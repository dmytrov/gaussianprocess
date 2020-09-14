# -*- coding: utf-8 -*-
from cgkit.all import *
import cgkit.bvhimport
import sys,time
import numpy as np
import pickle


# in-place camera
TargetCamera(
    #pos    = (-300,100,150),
    # pos    = (-300,100,00),
    pos    = (-150,100,300),
    target = (-0,100,0),
    up = (0,1,0)
)

redMat=GLMaterial("redstuff",ambient=(1.0,0.0,0.0),diffuse=(1.0,0.0,0.0),emission=(1.0,0.0,0.0))
greyMat=GLMaterial("greystuff",ambient=(.5,0.5,0.5),diffuse=(.5,0.5,0.5))

class JointBoneTransform(Component):
    """Compute the bone-local coordinate system from the end-joint of the bone. Start-joint is parent anyway"""

    def __init__(self, name="JointBoneTransform", auto_insert=True):
        Component.__init__(self, name=name, auto_insert=auto_insert)

        # Create the input slots
        self.input_slot = Mat4Slot()
        # Create the output slot
        self.output_slot = ProceduralMat4Slot(self.computeOutput)

        # Add the slots to the component
        self.addSlot("input", self.input_slot)
        self.addSlot("output", self.output_slot)

        # Set up slot dependencies
        self.input_slot.addDependent(self.output_slot)

    def computeOutput(self):

        # z-axis in the direction of the bone, towards the child joint
        #print(self.input)
        bz=vec3(list(self.input[3])[:3])
        #if bz.length() == 0:
        #    bz = vec3(1, 1, 1)
        bz/=bz.length()

        # determine y-axis as the projection of one of the joint axes into the plane perpendicular to bz
        # we choose that axis whose projection is longest, this should increase numerical stability.
        maxProjection=0.0
        by=vec3(0.0)
        for ycand in range(3):
            by1=vec3(list(self.input[ycand])[:3])
            by1/=by1.length()
            by1-=(by1*bz)*bz
            by1len=by1.length()
            if by1len>maxProjection:
                maxProjection=by1len
                by=by1/by1len
        # x axis by cross product
        bx=by.cross(bz)

        boneTrafo=mat4(1.0)
        boneTrafo[0]=vec4(bx)
        boneTrafo[1]=vec4(by)
        boneTrafo[2]=vec4(bz)
        boneTrafo[3]=self.input[3]/2
        boneTrafo[3,3]=1.0
  
        return boneTrafo
            
    
    # Create value attributes
    exec(slotPropertyCode("input"))
    exec(slotPropertyCode("output"))



class animatedBVHReader(cgkit.bvhimport.BVHReader):

    def __init__(self,filename,pickleFilename=None):
        cgkit.bvhimport.BVHReader.__init__(self,filename)
        if pickleFilename is None: self.pickleFilename=self.filename[:-3]+"pkl"
        else: self.pickleFilename=pickleFilename
        self.jointSpheres=[]
        self.boneCylinders=[]
        self.boneTransforms=[]
        self.jointMap=dict()
        self.startFrame=0
        self.endFrame=100000
        self.curFrame=0

    def createSkeleton(self, node, parent=None):
        """Create skeleton, and skin around it"""
        cgkit.bvhimport.BVHReader.createSkeleton(self,node,parent)
        node.joint.geom = cgkit.joint.JointGeom(node.joint, radius=0.0)

        nnl=node.name.lower()
        if parent is not None:
            pname=parent.name.lower()
        else: pname="none"

        # some special node geometries
        mat=greyMat
        jointRadius=5
        boneRadius=3
        scale=vec3(1.0)
            
        if nnl.find("pelvis")>=0: mat=redMat

        if nnl.find("end")>=0:
            if pname.find("head")>=0.0:
                jointRadius*=2
                scale=vec3(1.0,1.4,1.0)
            else: jointRadius=0.05

        if nnl.find("finger")>=0 or pname.find("finger")>=0:
            jointRadius/=1.4
            boneRadius/=1.4

        if nnl.find("spine")>=0 or pname.find("spine")>0:
            boneRadius*=2

        
        jointSphere=Sphere(node.name+".sphere",radius=jointRadius,parent=node.joint,material=mat,scale=scale)
        
        if not node.isRoot():
            length=node.joint.pos.length()
            bname=node.joint.parent.name+"-"+node.joint.name+".bone"
            boneCylinder=CCylinder(bname,radius=boneRadius,length=length,parent=node.joint.parent)
            bT=JointBoneTransform(bname)
            node.joint.transform_slot.connect(bT.input_slot)
            bT.output_slot.connect(boneCylinder.transform_slot)
            self.boneCylinders+=[boneCylinder]
            self.boneTransforms+=[bT]
            
        self.jointSpheres+=[jointSphere]
        self.jointMap[node.name]=node
       
        #self.identityTransforms+=[idTraf]

    def _iterNode(self,node,firstIdx=0):
        """Iterate through nodes to build the channel list, and the list of channel indexes (keep only positions if we're at the root node)"""
        channelList=[]
        channelIndexes=[]
        name=node.name
        for ch in node.channels:
            if ch[1:]=="rotation" or ch[1:]=="position":
                channelList+=[name+":"+ch]
                channelIndexes+=[firstIdx]
                
            firstIdx+=1

        for child in node.children:
            clc,cpc,firstIdx=self._iterNode(child,firstIdx)
            channelList+=clc
            channelIndexes+=cpc
            
        return channelList,channelIndexes,firstIdx
            

    def onMotion(self,frames,dt):
        # check if there is pickled data for this bvh
        try:
            chnames,self.pickledMotionData=self.unpickleData()
            self.pickledMotDat=list(self.pickledMotionData)
            frames=len(self.pickledMotionData)
        except:
            chnames=None
            self.pickledMotionData=None
        
        cgkit.bvhimport.BVHReader.onMotion(self,frames,dt)
        self.channelNames,self.channelIndexes,fi=self._iterNode(self.root)


        if chnames is not None and chnames!=self.channelNames: raise ValueError("BVH and pkl files have differing channels names -- please check")

        self.motionData=[]
        self.pelvisPosIdx=[]
        for chn,i in zip(self.channelNames,range(len(self.channelNames))):
            chn=chn.lower()
            if chn.find("pelvis")>=0 and chn.find("xposit")>=0 or (chn.find("pelvis")<0 and chn.find("posit")>=0) : self.pelvisPosIdx+=[i]
    
    def onFrame(self,values):
        
        if self.pickledMotionData is not None:
            if len(self.pickledMotionData)>0:
                values=self.pickledMotionData[0]
                self.pickledMotionData=self.pickledMotionData[1:]
            else: return
        for ppi in self.pelvisPosIdx: values[ppi]=0.0 # stop pelvis. makes the avatar walk in place
                
        cgkit.bvhimport.BVHReader.onFrame(self,values)
        self.motionData+=[np.array(values)[self.channelIndexes].reshape((-1,1))]

    def pickleData(self):
        motdat=np.concatenate(self.motionData,axis=1).T[self.startFrame:self.endFrame+1]
        md=[map(float,list(timestep)) for timestep in motdat]
        pf=open(self.pickleFilename,"w")
        pickle.dump({"channelNames":self.channelNames,"motionData":md},pf)
        pf.close()

    def unpickleData(self):
        pf=open(self.pickleFilename)
        pdat=pickle.load(pf)
        if not ("channelNames" in pdat and "motionData" in pdat): raise ValueError("pickle file "+self.pickleFilename+" does not seem to contain motion data")
        return pdat["channelNames"],pdat["motionData"]        
        pf.close()

    def setStartEndFrame(self,keyEvent):
        if keyEvent.key=="s" and self.endFrame>self.curFrame:
            self.startFrame=self.curFrame
            print("Starting at",self.curFrame)
        if keyEvent.key=="e":
            self.endFrame=self.curFrame
            print("End at",self.curFrame)

    def stepFrame(self):
        self.curFrame+=1
        print("At frame",self.curFrame,"of",len(self.motionData))
        if not(hasattr(self,"pastVals")): self.pastVals=[]
        self.pastVals+=[self.jointMap["Bip001_R_Toe0"].vtz.output_slot.getValue()]
        self.pastVals=self.pastVals[-3:]
        if len(self.pastVals)==3 and self.pastVals[0]>self.pastVals[1] and self.pastVals[1]<self.pastVals[2]:
            print("MINIMUM")
        
        #time.sleep(0.2)
        if self.curFrame>len(self.motionData):
            self.pickleData()
            sys.exit()
        

        


fn=sys.argv[-1]
pfn=None

if fn[-3:]=="pkl":
    pfn=fn
    fn=sys.argv[-2]

print(fn,pfn)
print(sys.argv)

if fn[-3:]!="pkl" and fn[-3:]!="bvh":
    fn="testBVH/DanicaMapped_HappyWalk01.bvh"
    pfn="testBVH/DanicaMapped_HappyWalk01.pkl"

print("DOING",fn)

mbi=animatedBVHReader(fn,pfn)
mbi.read()

print(mbi.jointMap.keys())


eventmanager.eventManager().connect(cgkit.events.STEP_FRAME,mbi.stepFrame)
eventmanager.eventManager().connect(cgkit.events.KEY_PRESS,mbi.setStartEndFrame)

