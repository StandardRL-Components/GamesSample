import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T09:49:36.368221
# Source Brief: brief_00101.md
# Brief Index: 101
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Control two robotic arms to catch components from a conveyor belt and assemble a structure. "
        "Time your movements to match the component size and build as high as you can."
    )
    user_guide = (
        "Controls: Use ←→ to adjust the left arm. Use 'space' and 'shift' to adjust the right arm. "
        "Catch the components between the arms to build a tower."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        
        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_CONVEYOR = (25, 30, 45)
        self.COLOR_ARM = (200, 200, 220)
        self.COLOR_ARM_GLOW = (150, 150, 255)
        self.COLOR_COMPONENT = (0, 150, 255)
        self.COLOR_COMPONENT_GLOW = (100, 200, 255)
        self.COLOR_STRUCTURE = (0, 100, 200)
        self.COLOR_SUCCESS = (0, 255, 150)
        self.COLOR_FAIL = (255, 50, 50)
        self.COLOR_TEXT = (220, 220, 220)

        # Game Parameters
        self.MAX_STEPS = 3000
        self.MAX_MISALIGNMENTS = 3
        self.WIN_SCORE = 150
        self.ASSEMBLY_LINE_X = 150
        self.CONVEYOR_Y = 280
        self.COMPONENT_SIZE = 30
        self.INITIAL_CONVEYOR_SPEED = 1.5
        self.CONVEYOR_SPEED_INCREASE = 0.05
        self.INITIAL_SPAWN_INTERVAL = 120
        self.MIN_SPAWN_INTERVAL = 45

        # Arm Parameters
        self.ARM_Y_START = 100
        self.ARM_Y_END = self.CONVEYOR_Y
        self.ARM_1_CENTER = self.ASSEMBLY_LINE_X - 50
        self.ARM_2_CENTER = self.ASSEMBLY_LINE_X + 50
        self.ARM_1_AMPLITUDE = 40
        self.ARM_2_AMPLITUDE = 40
        self.ARM_1_FREQ = 0.04
        self.ARM_2_FREQ = 0.045 # Slightly different for more challenge
        self.PHASE_SHIFT_AMOUNT = 0.15

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_large = pygame.font.SysFont("Consolas", 30)
            self.font_small = pygame.font.SysFont("Consolas", 20)
        except pygame.error:
            self.font_large = pygame.font.SysFont(None, 40)
            self.font_small = pygame.font.SysFont(None, 28)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.misalignments = 0
        self.game_over = False
        
        self.arm1_phase = 0.0
        self.arm2_phase = 0.0
        self.arm1_x = 0.0
        self.arm2_x = 0.0

        self.components = []
        self.assembled_structure = []
        self.particles = []
        
        self.conveyor_speed = self.INITIAL_CONVEYOR_SPEED
        self.spawn_interval = self.INITIAL_SPAWN_INTERVAL
        self.steps_until_next_spawn = self.INITIAL_SPAWN_INTERVAL // 2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.misalignments = 0
        self.game_over = False
        
        self.arm1_phase = self.np_random.uniform(-math.pi, math.pi)
        self.arm2_phase = self.np_random.uniform(-math.pi, math.pi)

        self.components = []
        self.assembled_structure = []
        self.particles = []

        self.conveyor_speed = self.INITIAL_CONVEYOR_SPEED
        self.spawn_interval = self.INITIAL_SPAWN_INTERVAL
        self.steps_until_next_spawn = self.INITIAL_SPAWN_INTERVAL // 2

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.1  # Survival reward

        self._handle_action(action)
        self._update_state()
        
        assembly_reward, misalignment_penalty = self._check_assembly()
        reward += assembly_reward + misalignment_penalty
        
        terminated = False
        truncated = False
        if self.score >= self.WIN_SCORE:
            reward += 100.0
            terminated = True
        elif self.misalignments >= self.MAX_MISALIGNMENTS:
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True # Use truncated for time limit
        
        self.game_over = terminated or truncated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Arm 1 Control (Left/Right)
        if movement == 3:  # Left
            self.arm1_phase -= self.PHASE_SHIFT_AMOUNT
        elif movement == 4:  # Right
            self.arm1_phase += self.PHASE_SHIFT_AMOUNT

        # Arm 2 Control (Space/Shift)
        if space_held:
            self.arm2_phase -= self.PHASE_SHIFT_AMOUNT # Retard
        if shift_held:
            self.arm2_phase += self.PHASE_SHIFT_AMOUNT # Advance
            
    def _update_state(self):
        # Update arm positions
        self.arm1_x = self.ARM_1_CENTER + self.ARM_1_AMPLITUDE * math.sin(self.ARM_1_FREQ * self.steps + self.arm1_phase)
        self.arm2_x = self.ARM_2_CENTER + self.ARM_2_AMPLITUDE * math.sin(self.ARM_2_FREQ * self.steps + self.arm2_phase)

        # Update difficulty
        self.conveyor_speed = self.INITIAL_CONVEYOR_SPEED + (self.steps // 100) * self.CONVEYOR_SPEED_INCREASE
        self.spawn_interval = max(self.MIN_SPAWN_INTERVAL, self.INITIAL_SPAWN_INTERVAL - (self.steps // 150) * 5)

        # Spawn new components
        self.steps_until_next_spawn -= 1
        if self.steps_until_next_spawn <= 0:
            self.components.append(pygame.Vector2(self.WIDTH + self.COMPONENT_SIZE, self.CONVEYOR_Y - self.COMPONENT_SIZE))
            self.steps_until_next_spawn = self.spawn_interval + self.np_random.integers(-10, 10)

        # Update components
        for comp in self.components:
            comp.x -= self.conveyor_speed
        
        # Remove off-screen components (but don't penalize)
        self.components = [comp for comp in self.components if comp.x > -self.COMPONENT_SIZE]

        # Update particles
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _check_assembly(self):
        reward = 0
        penalty = 0
        
        components_to_remove = []
        for i, comp in enumerate(self.components):
            comp_center_x = comp.x + self.COMPONENT_SIZE / 2
            
            # Check if component is in the assembly zone
            if self.ASSEMBLY_LINE_X - self.conveyor_speed < comp_center_x <= self.ASSEMBLY_LINE_X:
                components_to_remove.append(i)
                
                arm_dist = abs(self.arm2_x - self.arm1_x)
                
                # Success condition
                is_captured = (min(self.arm1_x, self.arm2_x) < comp_center_x < max(self.arm1_x, self.arm2_x))
                is_correct_width = abs(arm_dist - self.COMPONENT_SIZE) < 8 # Tolerance

                if is_captured and is_correct_width:
                    # SFX: Success chime
                    self.score += 1
                    reward += 1.0
                    
                    structure_y = self.CONVEYOR_Y - self.COMPONENT_SIZE - len(self.assembled_structure) * self.COMPONENT_SIZE
                    self.assembled_structure.append(pygame.Rect(self.ASSEMBLY_LINE_X - self.COMPONENT_SIZE/2, structure_y, self.COMPONENT_SIZE, self.COMPONENT_SIZE))
                    self._create_particles(pygame.Vector2(self.ASSEMBLY_LINE_X, self.CONVEYOR_Y - self.COMPONENT_SIZE/2), 20, self.COLOR_SUCCESS)
                else:
                    # SFX: Failure buzz
                    self.misalignments += 1
                    penalty -= 5.0
                    self._create_particles(pygame.Vector2(comp_center_x, comp.y + self.COMPONENT_SIZE/2), 40, self.COLOR_FAIL)

        # Remove processed components
        for i in sorted(components_to_remove, reverse=True):
            del self.components[i]

        return reward, penalty

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': life, 'color': color})

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "misalignments": self.misalignments,
        }

    def _render_all(self):
        # Background
        self.screen.fill(self.COLOR_BG)
        pygame.draw.rect(self.screen, self.COLOR_CONVEYOR, (0, self.CONVEYOR_Y, self.WIDTH, self.HEIGHT - self.CONVEYOR_Y))
        pygame.draw.line(self.screen, (50, 60, 80), (self.ASSEMBLY_LINE_X, 0), (self.ASSEMBLY_LINE_X, self.HEIGHT), 1)

        # Assembled Structure
        for block in self.assembled_structure:
            pygame.draw.rect(self.screen, self.COLOR_STRUCTURE, block)
            pygame.draw.rect(self.screen, self.COLOR_COMPONENT, block, 2)

        # Components
        for comp in self.components:
            rect = pygame.Rect(comp.x, comp.y, self.COMPONENT_SIZE, self.COMPONENT_SIZE)
            pygame.gfxdraw.box(self.screen, rect, (*self.COLOR_COMPONENT_GLOW, 50))
            pygame.draw.rect(self.screen, self.COLOR_COMPONENT, rect, 0, border_radius=3)
            pygame.draw.rect(self.screen, (255,255,255), rect, 1, border_radius=3)

        # Arms
        self._render_arm(self.arm1_x)
        self._render_arm(self.arm2_x)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30.0))
            color = (*p['color'], alpha)
            size = max(1, int(4 * (p['life'] / 30.0)))
            if p['color'] == self.COLOR_FAIL: # Sparks
                pygame.draw.line(self.screen, color, p['pos'], p['pos'] + p['vel']*0.5, size)
            else: # Glow
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), size, color)

        # UI
        self._render_ui()

    def _render_arm(self, x_pos):
        x = int(x_pos)
        # Glow effect
        for i in range(5, 0, -1):
            alpha = 80 - i * 15
            pygame.gfxdraw.vline(self.screen, x, self.ARM_Y_START, self.ARM_Y_END, (*self.COLOR_ARM_GLOW, alpha))
            pygame.gfxdraw.vline(self.screen, x-i, self.ARM_Y_START, self.ARM_Y_END, (*self.COLOR_ARM_GLOW, alpha))
            pygame.gfxdraw.vline(self.screen, x+i, self.ARM_Y_START, self.ARM_Y_END, (*self.COLOR_ARM_GLOW, alpha))
        
        # Core line
        pygame.gfxdraw.vline(self.screen, x, self.ARM_Y_START, self.ARM_Y_END, self.COLOR_ARM)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = max(0, self.MAX_STEPS - self.steps)
        time_text = self.font_large.render(f"{time_left:04d}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH // 2 - time_text.get_width() // 2, 10))

        # Misalignments
        for i in range(self.MAX_MISALIGNMENTS):
            pos = (self.WIDTH - 40 - i * 35, 25)
            if i < self.misalignments:
                color = self.COLOR_FAIL
                pygame.draw.line(self.screen, color, (pos[0]-10, pos[1]-10), (pos[0]+10, pos[1]+10), 4)
                pygame.draw.line(self.screen, color, (pos[0]-10, pos[1]+10), (pos[0]+10, pos[1]-10), 4)
            else:
                color = (80, 80, 80)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, color)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, (*color, 50))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
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

if __name__ == '__main__':
    # This block allows you to play the game manually
    # For this to work, you must comment out the line: os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    env = GameEnv(render_mode="rgb_array")
    
    # To run the validation, you can uncomment the following line:
    # env.validate_implementation()
    
    obs, info = env.reset()
    
    # The following code will not display a window because of the "dummy" video driver.
    # It is intended for running the environment in a headless mode.
    # If you want to see the game, you need to handle rendering differently,
    # for example by saving frames or using a different render_mode if available.
    print("Running in headless mode. No window will be displayed.")
    
    # Example of a simple agent loop
    terminated = False
    truncated = False
    total_reward = 0
    step_count = 0

    while not terminated and not truncated:
        action = env.action_space.sample() # Random agent
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        if step_count % 100 == 0:
            print(f"Step: {step_count}, Score: {info['score']}, Reward: {total_reward:.2f}")

    print(f"\nGame Over!")
    print(f"Final Score: {info['score']}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Total Steps: {info['steps']}")

    env.close()