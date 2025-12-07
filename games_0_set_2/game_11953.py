import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import random
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper class for a single robotic arm
class _Arm:
    def __init__(self, x_pos, base_speed, base_reach, base_length, max_segments):
        self.x = x_pos
        self.segments = 1
        self.base_speed = base_speed
        self.base_reach_radius = base_reach
        self.base_length = base_length
        self.max_segments = max_segments
        self.update_stats()

    def update_stats(self):
        # Speed decreases by 10% of base for each segment *after the first*.
        # Use max to prevent speed from becoming too low.
        self.speed = max(self.base_speed * 0.1, self.base_speed * (1.0 - 0.10 * (self.segments - 1)))
        # Reach increases by 5% of base for each segment *after the first*.
        self.reach_radius = self.base_reach_radius * (1.0 + 0.05 * (self.segments - 1))
        # Arm length increases with segments.
        self.length = self.base_length + 20 * (self.segments - 1)

    def extend(self):
        if self.segments < self.max_segments:
            self.segments += 1
            self.update_stats()
            return True
        return False

    def get_gripper_pos(self):
        return pygame.Vector2(self.x, self.length)

# Helper class for visual effect particles
class _Particle:
    def __init__(self, x, y, color):
        self.pos = pygame.Vector2(x, y)
        self.color = color
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1.5, 4)
        self.vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        self.total_lifetime = random.randint(20, 40)
        self.lifetime = self.total_lifetime
        self.radius = random.uniform(4, 7)

    def update(self):
        self.pos += self.vel
        self.lifetime -= 1
        self.vel *= 0.95 # friction
        self.radius = max(0, self.radius - 0.15)

    def draw(self, surface):
        if self.lifetime > 0 and self.radius > 0:
            alpha = int(255 * (self.lifetime / self.total_lifetime))
            r, g, b = self.color
            try:
                # Use a temporary surface for blending alpha correctly
                temp_surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(temp_surf, int(self.radius), int(self.radius), int(self.radius), (r, g, b, alpha))
                surface.blit(temp_surf, (int(self.pos.x - self.radius), int(self.pos.y - self.radius)))
            except pygame.error:
                # Can happen if radius becomes too small for a surface
                pass

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control two robotic arms to collect all the floating objects before time runs out. "
        "Move, extend, and grasp to clear the play area and maximize your score."
    )
    user_guide = (
        "Controls: Use ←→ to move the selected arm. Use ↑↓ to switch between arms. "
        "Press space to grasp objects and shift to extend the arm."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_OBJECTS = 50
    TIME_LIMIT_SECONDS = 60
    # The brief requests 6000 steps, implying 100Hz.
    # We will use this for the step limit.
    MAX_STEPS = 6000

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_BOUNDARY = (50, 60, 80)
    COLOR_ARM = (220, 220, 240)
    COLOR_ARM_SELECTED_GLOW = (255, 255, 100)
    COLOR_OBJECT = (0, 255, 150)
    COLOR_TIMER_BAR = (220, 50, 50)
    COLOR_UI_TEXT = (255, 255, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_indicator = pygame.font.SysFont("Consolas", 18)

        # Game state variables
        self.arms = []
        self.objects = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.selected_arm_idx = 0
        self.prev_action = np.array([0, 0, 0])

        # Play area definition
        self.PLAY_AREA_CENTER = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 20)
        self.PLAY_AREA_RADIUS = 180
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.selected_arm_idx = 0
        self.prev_action = np.array([0, 0, 0])
        self.particles.clear()
        
        # Initialize arms
        self.arms.clear()
        arm1 = _Arm(x_pos=self.SCREEN_WIDTH * 0.33, base_speed=2.5, base_reach=20, base_length=60, max_segments=8)
        arm2 = _Arm(x_pos=self.SCREEN_WIDTH * 0.66, base_speed=2.5, base_reach=20, base_length=60, max_segments=8)
        self.arms.extend([arm1, arm2])

        # Initialize objects
        self.objects.clear()
        while len(self.objects) < self.TARGET_OBJECTS:
            angle = random.uniform(0, 2 * math.pi)
            # Spawn closer to center initially for easier start
            dist = random.uniform(0, self.PLAY_AREA_RADIUS - 15) 
            x = self.PLAY_AREA_CENTER[0] + math.cos(angle) * dist
            y = self.PLAY_AREA_CENTER[1] + math.sin(angle) * dist
            self.objects.append(pygame.Vector2(x, y))

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing and return the final state
            obs = self._get_observation()
            return obs, 0.0, True, False, self._get_info()
            
        self.steps += 1
        reward = 0.0
        
        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        prev_movement, prev_space_held, prev_shift_held = self.prev_action[0], self.prev_action[1] == 1, self.prev_action[2] == 1

        selected_arm = self.arms[self.selected_arm_idx]

        # --- Handle Actions ---
        # 1. Arm Selection (Discrete Press)
        if movement == 1 and prev_movement != 1: # Up -> Previous arm
            self.selected_arm_idx = (self.selected_arm_idx - 1 + len(self.arms)) % len(self.arms)
        elif movement == 2 and prev_movement != 2: # Down -> Next arm
            self.selected_arm_idx = (self.selected_arm_idx + 1) % len(self.arms)

        # 2. Arm Movement (Continuous Hold)
        if movement == 3: # Left
            selected_arm.x = max(0, selected_arm.x - selected_arm.speed)
        elif movement == 4: # Right
            selected_arm.x = min(self.SCREEN_WIDTH, selected_arm.x + selected_arm.speed)

        # 3. Extend Arm (Discrete Press)
        if shift_held and not prev_shift_held:
            if selected_arm.extend():
                pass

        # 4. Grasp (Discrete Press)
        if space_held and not prev_space_held:
            gripper_pos = selected_arm.get_gripper_pos()
            # Iterate backwards to safely remove items
            for i in range(len(self.objects) - 1, -1, -1):
                obj_pos = self.objects[i]
                if gripper_pos.distance_to(obj_pos) < selected_arm.reach_radius:
                    self.objects.pop(i)
                    self.score += 1
                    reward += 0.1
                    # Spawn particles for visual feedback
                    for _ in range(10):
                        self.particles.append(_Particle(obj_pos.x, obj_pos.y, self.COLOR_OBJECT))
                    # Can only grab one object per press for simplicity
                    break

        self.prev_action = action
        self._update_particles()
        
        # --- Check Termination ---
        win = self.score >= self.TARGET_OBJECTS
        timeout = self.steps >= self.MAX_STEPS
        terminated = win or timeout

        if terminated:
            self.game_over = True
            if win:
                reward += 100.0 # Victory bonus
            else: # Timeout
                reward -= 100.0 # Failure penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _update_particles(self):
        # Update and remove dead particles
        self.particles = [p for p in self.particles if p.lifetime > 0]
        for p in self.particles:
            p.update()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining_steps": self.MAX_STEPS - self.steps,
            "selected_arm": self.selected_arm_idx,
            "arm_segments": [arm.segments for arm in self.arms]
        }

    def _render_game(self):
        # 1. Draw play area boundary
        pygame.gfxdraw.aacircle(self.screen, self.PLAY_AREA_CENTER[0], self.PLAY_AREA_CENTER[1], self.PLAY_AREA_RADIUS, self.COLOR_BOUNDARY)

        # 2. Draw particles
        for p in self.particles:
            p.draw(self.screen)

        # 3. Draw objects
        for obj_pos in self.objects:
            pygame.gfxdraw.filled_circle(self.screen, int(obj_pos.x), int(obj_pos.y), 5, self.COLOR_OBJECT)
            pygame.gfxdraw.aacircle(self.screen, int(obj_pos.x), int(obj_pos.y), 5, self.COLOR_OBJECT)

        # 4. Draw arms
        for i, arm in enumerate(self.arms):
            start_pos = (int(arm.x), 0)
            gripper_pos = (int(arm.x), int(arm.length))
            
            # Draw arm shaft
            pygame.draw.line(self.screen, self.COLOR_ARM, start_pos, gripper_pos, 4)
            
            # Draw gripper
            pygame.gfxdraw.filled_circle(self.screen, gripper_pos[0], gripper_pos[1], 8, self.COLOR_ARM)
            pygame.gfxdraw.aacircle(self.screen, gripper_pos[0], gripper_pos[1], 8, self.COLOR_ARM)
            
            # Highlight selected arm
            if i == self.selected_arm_idx:
                # Draw reach radius indicator
                pygame.gfxdraw.aacircle(self.screen, gripper_pos[0], gripper_pos[1], int(arm.reach_radius), (*self.COLOR_ARM_SELECTED_GLOW, 100))
                # Draw glow effect
                for j in range(4):
                    alpha = 80 - j * 20
                    radius = 10 + j * 3
                    pygame.gfxdraw.filled_circle(self.screen, gripper_pos[0], gripper_pos[1], radius, (*self.COLOR_ARM_SELECTED_GLOW, alpha))

    def _render_ui(self):
        # 1. Draw timer bar
        time_ratio = max(0, (self.MAX_STEPS - self.steps) / self.MAX_STEPS)
        bar_width = int(self.SCREEN_WIDTH * time_ratio)
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR, (0, 0, bar_width, 8))
        
        # 2. Draw score
        score_text = self.font_ui.render(f"OBJECTS: {self.score}/{self.TARGET_OBJECTS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 15))

        # 3. Draw selected arm indicator
        for i, arm in enumerate(self.arms):
            indicator_text = f"ARM {i+1}"
            color = self.COLOR_ARM_SELECTED_GLOW if i == self.selected_arm_idx else self.COLOR_UI_TEXT
            text_surface = self.font_indicator.render(indicator_text, True, color)
            
            # Position indicators based on arm's initial position
            text_x = self.SCREEN_WIDTH * (0.33 if i == 0 else 0.66) - text_surface.get_width() / 2
            self.screen.blit(text_surface, (int(text_x), 15))

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It is not part of the required environment, but is useful for development
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Robotic Arm Collector")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0.0
    
    while running:
        # --- Action mapping from keyboard to MultiDiscrete action space ---
        keys = pygame.key.get_pressed()
        
        # Movement (0=none, 1=up, 2=down, 3=left, 4=right)
        movement_action = 0 # none
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        
        # Space button (0=released, 1=held)
        space_action = 1 if keys[pygame.K_SPACE] else 0
        
        # Shift button (0=released, 1=held)
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([movement_action, space_action, shift_action])
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- RESET ---")
                obs, info = env.reset()
                total_reward = 0.0

        # --- Rendering ---
        # The observation is already a rendered frame
        # Transpose it back for pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            print("Press 'R' to reset.")
            
        clock.tick(60) # Limit frame rate for human playability
        
    env.close()