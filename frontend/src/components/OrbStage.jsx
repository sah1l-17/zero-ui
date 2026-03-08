import Orb from './Orb';

export default function OrbStage({ speaking }) {
  return (
    <div className="orbStage">
      <div className="orbLabel" aria-hidden="true">
        ASSISTANT
      </div>
      <Orb
        hoverIntensity={0.30}
        rotateOnHover
        hue={0}
        forceHoverState={speaking}
        backgroundColor="#0b0f14"
      />
    </div>
  );
}
