import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T09:55:06.589643
# Source Brief: brief_00157.md
# Brief Index: 157
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper class for Robotic Arms
class RoboticArm:
    def __init__(self, base_pos, speed, reach, color, arm_id):
        self.base_pos = pygame.Vector2(base_pos)
        self.pos = pygame.Vector2(base_pos)
        self.speed = speed
        self.reach = reach
        self.color = color
        self.arm_id = arm_id
        self.tool = None
        self.available_tools = ['gripper', 'welder', 'bolt_driver']
        self.part_held = False
        self.target_angle = 0
        self.current_angle = 0

    def update(self, move_dir):
        velocity = move_dir * self.speed
        self.pos += velocity
        
        # Clamp position to reach
        from_base = self.pos - self.base_pos
        dist = from_base.length()
        if dist > self.reach:
            self.pos = self.base_pos + from_base.normalize() * self.reach
        
        if from_base.length() > 0:
            self.target_angle = from_base.angle_to(pygame.Vector2(1, 0))
        
        # Smooth rotation
        angle_diff = (self.target_angle - self.current_angle + 180) % 360 - 180
        self.current_angle += angle_diff * 0.2

    def transform(self):
        if self.available_tools:
            self.tool = self.available_tools.pop(0)
            return True
        self.tool = None # Cycle back to none if all are used
        return False

    def draw(self, surface, is_active):
        # Draw reach circle
        pygame.gfxdraw.aacircle(surface, int(self.base_pos.x), int(self.base_pos.y), int(self.reach), (50, 60, 70))

        # Draw arm body
        arm_width = 12
        p1 = self.base_pos
        p2 = self.pos
        
        angle_rad = math.radians(self.current_angle)
        dx = math.sin(angle_rad) * arm_width / 2
        dy = math.cos(angle_rad) * arm_width / 2

        points = [
            (p1.x - dx, p1.y + dy),
            (p1.x + dx, p1.y - dy),
            (p2.x + dx, p2.y - dy),
            (p2.x - dx, p2.y + dy),
        ]
        
        arm_color = (100, 120, 130)
        pygame.gfxdraw.aapolygon(surface, points, arm_color)
        pygame.gfxdraw.filled_polygon(surface, points, arm_color)

        # Draw joints
        pygame.gfxdraw.filled_circle(surface, int(self.base_pos.x), int(self.base_pos.y), 10, (180, 180, 190))
        pygame.gfxdraw.aacircle(surface, int(self.base_pos.x), int(self.base_pos.y), 10, (180, 180, 190))

        # Draw end effector (head)
        head_color = self.color
        if is_active:
            head_color = (255, 255, 0) # Active color
            # Glow effect for active arm
            for i in range(5):
                alpha = 80 - i * 15
                pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), 12 + i * 2, (*head_color, alpha))

        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), 10, head_color)
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), 10, head_color)

        # Draw tool icon
        if self.tool:
            self._draw_tool(surface, self.pos, self.tool)

    def _draw_tool(self, surface, pos, tool):
        if tool == 'gripper':
            pygame.draw.line(surface, (200, 200, 200), (pos.x - 8, pos.y - 8), (pos.x - 14, pos.y - 14), 3)
            pygame.draw.line(surface, (200, 200, 200), (pos.x + 8, pos.y - 8), (pos.x + 14, pos.y - 14), 3)
        elif tool == 'welder':
            pygame.gfxdraw.filled_trigon(surface, int(pos.x), int(pos.y - 15), int(pos.x-5), int(pos.y-8), int(pos.x+5), int(pos.y-8), (255, 150, 0))
        elif tool == 'bolt_driver':
            pygame.draw.rect(surface, (180, 180, 200), (pos.x - 5, pos.y - 15, 10, 10))

# Helper class for Particles
class Particle:
    def __init__(self, x, y, color):
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(random.uniform(-2, 2), random.uniform(-2, 2))
        self.color = color
        self.lifespan = random.randint(10, 25)
        self.radius = random.uniform(2, 5)

    def update(self):
        self.pos += self.vel
        self.vel *= 0.95
        self.lifespan -= 1
        self.radius -= 0.1

    def draw(self, surface):
        if self.lifespan > 0 and self.radius > 0:
            alpha = int(255 * (self.lifespan / 25))
            color = (*self.color, alpha)
            pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), int(self.radius), color)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a team of robotic arms to assemble a component. Switch between arms, equip the right tools, and complete the assembly sequence before time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the active arm. Press space to switch between arms and shift to change the arm's tool."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1200 # Approx 40 seconds at 30fps

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_ARM_1 = (50, 150, 255) # Blue
        self.COLOR_ARM_2 = (255, 80, 80) # Red
        self.COLOR_ARM_3 = (80, 255, 80) # Green
        self.COLOR_PART = (150, 150, 160)
        self.COLOR_SLOT = (80, 80, 90)
        self.COLOR_WELD_SPARK = (255, 180, 0)
        self.COLOR_SUCCESS = (0, 255, 128)

        # Game state variables
        self.arms = []
        self.active_arm_index = 0
        self.assembly_stage = 0
        self.assembly_targets = []
        self.part_pos = pygame.Vector2(0, 0)
        self.part_held_by = -1
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.prev_space_held = 0
        self.prev_shift_held = 0
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.active_arm_index = 0
        self.prev_space_held = 0
        self.prev_shift_held = 0
        self.part_held_by = -1

        # Assembly setup
        slot_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.assembly_targets = [
            None, # Set in loop
            slot_pos,
            slot_pos,
            slot_pos,
        ]
        self.part_pos = pygame.Vector2(self.np_random.uniform(100, self.WIDTH-100), self.np_random.uniform(250, self.HEIGHT-50))
        self.assembly_targets[0] = self.part_pos
        self.assembly_stage = 0 # 0:pickup, 1:place, 2:weld, 3:bolt

        # Initialize arms
        self.arms = [
            RoboticArm(base_pos=(100, 50), speed=3.0, reach=150, color=self.COLOR_ARM_1, arm_id=0),
            RoboticArm(base_pos=(self.WIDTH/2, 50), speed=2.25, reach=220, color=self.COLOR_ARM_2, arm_id=1),
            RoboticArm(base_pos=(self.WIDTH-100, 50), speed=3.75, reach=280, color=self.COLOR_ARM_3, arm_id=2),
        ]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement_idx = action[0]
        space_held = action[1]
        shift_held = action[2]
        
        reward = -0.001 # Small penalty for existing
        self.steps += 1
        
        # --- Action Processing ---
        active_arm = self.arms[self.active_arm_index]
        
        # Movement
        move_dir = pygame.Vector2(0, 0)
        if movement_idx == 1: move_dir.y = -1
        elif movement_idx == 2: move_dir.y = 1
        elif movement_idx == 3: move_dir.x = -1
        elif movement_idx == 4: move_dir.x = 1
        active_arm.update(move_dir)

        # Arm selection (on press)
        if space_held and not self.prev_space_held:
            self.active_arm_index = (self.active_arm_index + 1) % len(self.arms)
            # _sound_effect_: arm_select.wav
            reward -= 0.01 # Small cost to switch

        # Tool transformation (on press)
        if shift_held and not self.prev_shift_held:
            if active_arm.transform():
                # _sound_effect_: tool_transform.wav
                reward -= 0.05 # Cost to transform
            else:
                reward -= 0.1 # Penalty for failed transform

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        # --- Game Logic ---
        self._update_collisions()

        if self.part_held_by != -1:
            self.part_pos = self.arms[self.part_held_by].pos

        # Check for assembly task completion
        current_target = self.assembly_targets[self.assembly_stage]
        dist_to_target = active_arm.pos.distance_to(current_target)
        
        if dist_to_target < 20: # Proximity threshold
            if self.assembly_stage == 0 and active_arm.tool == 'gripper' and self.part_held_by == -1:
                self.part_held_by = self.active_arm_index
                self.assembly_stage += 1
                reward += 1.0
                self.score += 10
                # _sound_effect_: pickup.wav
            elif self.assembly_stage == 1 and active_arm.tool == 'gripper' and self.part_held_by == self.active_arm_index:
                self.part_held_by = -1 # Release part
                self.part_pos = self.assembly_targets[1] # Snap to slot
                self.assembly_stage += 1
                reward += 1.0
                self.score += 10
                # _sound_effect_: place.wav
            elif self.assembly_stage == 2 and active_arm.tool == 'welder':
                self.assembly_stage += 1
                reward += 1.0
                self.score += 10
                # _sound_effect_: weld.wav
                for _ in range(30): self.particles.append(Particle(active_arm.pos.x, active_arm.pos.y, self.COLOR_WELD_SPARK))
            elif self.assembly_stage == 3 and active_arm.tool == 'bolt_driver':
                self.assembly_stage += 1 # Win condition met
                reward += 1.0
                self.score += 10
                # _sound_effect_: bolt.wav

        # Update particles
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()
        
        # --- Termination and Final Reward ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.assembly_stage >= 4: # Win
                reward += 100
                self.score += 1000
                # _sound_effect_: victory.wav
            else: # Timeout
                reward -= 100
                # _sound_effect_: failure.wav
        
        truncated = False
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_collisions(self):
        for i in range(len(self.arms)):
            for j in range(i + 1, len(self.arms)):
                arm1 = self.arms[i]
                arm2 = self.arms[j]
                dist_vec = arm1.pos - arm2.pos
                dist = dist_vec.length()
                min_dist = 20 # Collision radius
                if dist < min_dist and dist > 0:
                    overlap = (min_dist - dist) / 2
                    push_vec = dist_vec.normalize() * overlap
                    arm1.pos += push_vec
                    arm2.pos -= push_vec
                    # Re-clamp to reach after push
                    from_base1 = arm1.pos - arm1.base_pos
                    if from_base1.length() > arm1.reach: arm1.pos = arm1.base_pos + from_base1.normalize() * arm1.reach
                    from_base2 = arm2.pos - arm2.base_pos
                    if from_base2.length() > arm2.reach: arm2.pos = arm2.base_pos + from_base2.normalize() * arm2.reach

    def _check_termination(self):
        return self.steps >= self.MAX_STEPS or self.assembly_stage >= 4
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw assembly slot
        slot_pos = self.assembly_targets[1]
        slot_size = 30
        slot_rect = pygame.Rect(slot_pos.x - slot_size/2, slot_pos.y - slot_size/2, slot_size, slot_size)
        pygame.draw.rect(self.screen, self.COLOR_SLOT, slot_rect, border_radius=4)
        if self.assembly_stage > 1: # Part placed
             pygame.draw.rect(self.screen, self.COLOR_PART, slot_rect.inflate(-8,-8), border_radius=2)
        if self.assembly_stage > 2: # Welded
             pygame.draw.rect(self.screen, self.COLOR_WELD_SPARK, slot_rect.inflate(-12,-12), 2, border_radius=1)
        if self.assembly_stage > 3: # Bolted
             pygame.draw.circle(self.screen, (200,200,210), slot_rect.center, 5)


        # Draw part if not held and not placed
        if self.part_held_by == -1 and self.assembly_stage < 2:
            part_size = 20
            part_rect = pygame.Rect(self.part_pos.x - part_size/2, self.part_pos.y - part_size/2, part_size, part_size)
            pygame.draw.rect(self.screen, self.COLOR_PART, part_rect, border_radius=2)

        # Draw arms
        for i, arm in enumerate(self.arms):
            arm.draw(self.screen, i == self.active_arm_index)
            if arm.part_held:
                part_size = 20
                part_rect = pygame.Rect(arm.pos.x - part_size/2, arm.pos.y - part_size/2, part_size, part_size)
                pygame.draw.rect(self.screen, self.COLOR_PART, part_rect, border_radius=2)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Timer bar
        time_ratio = (self.MAX_STEPS - self.steps) / self.MAX_STEPS
        bar_width = self.WIDTH * time_ratio
        pygame.draw.rect(self.screen, (200, 50, 50), (0, 0, self.WIDTH, 10))
        pygame.draw.rect(self.screen, (50, 200, 50), (0, 0, bar_width, 10))

        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, self.HEIGHT - 30))
        
        # Steps
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, self.HEIGHT - 30))

        # Assembly checklist
        tasks = ["PICK UP PART", "PLACE PART", "WELD PART", "BOLT PART"]
        for i, task in enumerate(tasks):
            color = self.COLOR_UI_TEXT
            prefix = "[ ]"
            if self.assembly_stage > i:
                color = self.COLOR_SUCCESS
                prefix = "[X]"
            elif self.assembly_stage == i:
                color = (255, 255, 0) # Current task
                prefix = "[>]"
            
            task_text = self.font_small.render(f"{prefix} {task}", True, color)
            self.screen.blit(task_text, (self.WIDTH - 180, 20 + i * 25))

        # Game Over/Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            if self.assembly_stage >= 4:
                msg = "ASSEMBLY COMPLETE"
                color = self.COLOR_SUCCESS
            else:
                msg = "TIME OUT"
                color = (255, 50, 50)
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "assembly_stage": self.assembly_stage,
            "active_arm": self.active_arm_index,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        self.reset()
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Robo-Assembler")
    clock = pygame.time.Clock()
    running = True

    movement = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
    space = 0 # 0=released, 1=held
    shift = 0 # 0=released, 1=held

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False

        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}, Steps: {info['steps']}")
            # Wait a bit before auto-resetting for human player
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(env.FPS)

    env.close()