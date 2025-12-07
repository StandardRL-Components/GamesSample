import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:29:18.509786
# Source Brief: brief_02199.md
# Brief Index: 2199
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls four robotic arms to sort
    colored blocks onto a rotating conveyor belt. The goal is to create triplets
    of same-colored blocks to score points within a 90-second time limit.
    This environment prioritizes visual quality and engaging gameplay.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right) - affects the selected arm.
    - actions[1]: Space button (0=released, 1=held) - drops a block from the selected arm.
    - actions[2]: Shift button (0=released, 1=held) - selects the next arm.

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control four robotic arms to sort colored blocks onto a rotating conveyor belt. "
        "Create triplets of same-colored blocks to score points before time runs out."
    )
    user_guide = (
        "Controls: Use ←→ to move the selected arm. Press space to drop a block and shift to select the next arm."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    GAME_DURATION_SECONDS = 90

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 50, 60)
    COLOR_BELT = (60, 70, 80)
    COLOR_BELT_RIM = (90, 100, 110)
    COLOR_ARM = (150, 160, 170)
    COLOR_ARM_SELECTED = (255, 255, 100)
    COLOR_ARM_BASE = (100, 110, 120)
    COLOR_TEXT = (220, 220, 230)
    BLOCK_COLORS = {
        "red": (255, 70, 70),
        "green": (70, 255, 70),
        "blue": (70, 130, 255)
    }
    BLOCK_GLOW_COLORS = {
        "red": (120, 20, 20),
        "green": (20, 120, 20),
        "blue": (20, 50, 120)
    }

    # Game Mechanics
    ARM_Y_POS = 60
    ARM_WIDTH = 10
    ARM_HEIGHT = 50
    ARM_BASE_SPEED = 0.4
    ARM_SPEED_MODIFIERS = [0.5, 1.0, 1.5, 2.0]
    ARM_FRICTION = 0.90
    ARM_X_LIMITS = (100, SCREEN_WIDTH - 100)
    BLOCK_SPAWN_DELAY = 30  # steps

    BELT_CENTER = (SCREEN_WIDTH // 2, SCREEN_HEIGHT + 100)
    BELT_RADIUS = 220
    BELT_ROTATION_SPEED = 0.005  # radians per step
    BLOCK_SIZE = 10
    MATCH_ANGLE_THRESHOLD = 0.4  # radians

    # Rewards
    REWARD_DROP = 0.1
    REWARD_MATCH = 10.0
    REWARD_BONUS = 50.0
    BONUS_SCORE_THRESHOLD = 500

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)
        
        # State variables are initialized in reset()
        self.arms = []
        self.belt_blocks = []
        self.particles = []
        self.belt_angle = 0.0
        self.selected_arm_idx = 0
        self.score = 0
        self.steps = 0
        self.timer = 0
        self.game_over = False
        self.bonus_awarded = False
        self.prev_space_held = False
        self.prev_shift_held = False
        
        # Using np.random.default_rng() for modern numpy
        self.np_random = np.random.default_rng()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.timer = self.FPS * self.GAME_DURATION_SECONDS
        self.game_over = False
        self.bonus_awarded = False
        
        self.belt_angle = 0.0
        self.belt_blocks = []
        self.particles = []
        
        self.selected_arm_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.arms = []
        num_arms = 4
        spacing = (self.ARM_X_LIMITS[1] - self.ARM_X_LIMITS[0]) / (num_arms - 1)
        for i in range(num_arms):
            arm = {
                "x": self.ARM_X_LIMITS[0] + i * spacing,
                "vx": 0.0,
                "held_block": None,
                "spawn_timer": 0,
                "base_x": self.ARM_X_LIMITS[0] + i * spacing
            }
            self.arms.append(arm)
            self._spawn_new_block(i)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # --- Handle Input ---
        # Switch arm on button press (not hold)
        if shift_held and not self.prev_shift_held:
            self.selected_arm_idx = (self.selected_arm_idx + 1) % len(self.arms)
            # sfx: arm_select_sound

        # Drop block on button press
        if space_held and not self.prev_space_held:
            arm = self.arms[self.selected_arm_idx]
            if arm["held_block"] and arm["spawn_timer"] == 0:
                self._drop_block(self.selected_arm_idx)
                reward += self.REWARD_DROP
                # sfx: block_drop_sound

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Update Game State ---
        self._update_arms(movement)
        self._update_belt()
        self._update_particles()
        
        # --- Check for Matches ---
        matches_found = self._check_matches()
        if matches_found > 0:
            reward += matches_found * self.REWARD_MATCH
            # sfx: match_success_sound

        # --- Score Bonus Reward ---
        if self.score >= self.BONUS_SCORE_THRESHOLD and not self.bonus_awarded:
            reward += self.REWARD_BONUS
            self.bonus_awarded = True
            # sfx: bonus_achieved_sound

        # --- Update Timers ---
        self.steps += 1
        self.timer -= 1
        
        terminated = self.timer <= 0
        if terminated:
            self.game_over = True

        return self._get_observation(), float(reward), terminated, False, self._get_info()

    def _update_arms(self, movement):
        # Update selected arm acceleration
        selected_arm = self.arms[self.selected_arm_idx]
        accel = 0
        if movement == 3:  # Left
            accel = -self.ARM_BASE_SPEED * self.ARM_SPEED_MODIFIERS[self.selected_arm_idx]
        elif movement == 4:  # Right
            accel = self.ARM_BASE_SPEED * self.ARM_SPEED_MODIFIERS[self.selected_arm_idx]
        
        selected_arm["vx"] += accel

        # Update all arms (physics and spawning)
        for i, arm in enumerate(self.arms):
            arm["vx"] *= self.ARM_FRICTION
            arm["x"] += arm["vx"]
            arm["x"] = np.clip(arm["x"], self.ARM_X_LIMITS[0], self.ARM_X_LIMITS[1])
            
            if arm["spawn_timer"] > 0:
                arm["spawn_timer"] -= 1
                if arm["spawn_timer"] == 0:
                    self._spawn_new_block(i)

    def _update_belt(self):
        self.belt_angle += self.BELT_ROTATION_SPEED
        if self.belt_angle > 2 * math.pi:
            self.belt_angle -= 2 * math.pi
    
    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1

    def _spawn_new_block(self, arm_idx):
        color_name = self.np_random.choice(list(self.BLOCK_COLORS.keys()))
        self.arms[arm_idx]["held_block"] = {
            "name": color_name,
            "color": self.BLOCK_COLORS[color_name],
            "glow": self.BLOCK_GLOW_COLORS[color_name]
        }
    
    def _drop_block(self, arm_idx):
        arm = self.arms[arm_idx]
        block = arm["held_block"]
        arm["held_block"] = None
        arm["spawn_timer"] = self.BLOCK_SPAWN_DELAY

        # Calculate drop position on belt
        drop_x, drop_y = arm["x"], self.ARM_Y_POS + self.ARM_HEIGHT
        dx = drop_x - self.BELT_CENTER[0]
        dy = drop_y - self.BELT_CENTER[1]
        
        angle = math.atan2(dy, dx) - self.belt_angle
        
        self.belt_blocks.append({
            "name": block["name"],
            "color": block["color"],
            "glow": block["glow"],
            "angle": angle
        })

    def _check_matches(self):
        if len(self.belt_blocks) < 3:
            return 0
        
        # Normalize angles to be positive
        for block in self.belt_blocks:
            block['angle'] = block['angle'] % (2 * math.pi)

        self.belt_blocks.sort(key=lambda b: b['angle'])
        
        matched_indices = set()
        num_matches = 0
        
        for i in range(len(self.belt_blocks)):
            i1 = i
            i2 = (i + 1) % len(self.belt_blocks)
            i3 = (i + 2) % len(self.belt_blocks)
            
            b1 = self.belt_blocks[i1]
            b2 = self.belt_blocks[i2]
            b3 = self.belt_blocks[i3]
            
            # Check for same color
            if b1["name"] == b2["name"] == b3["name"]:
                # Check for adjacency
                angle_diff1 = abs(b2['angle'] - b1['angle'])
                if i2 < i1: angle_diff1 = abs(b2['angle'] + 2*math.pi - b1['angle'])

                angle_diff2 = abs(b3['angle'] - b2['angle'])
                if i3 < i2: angle_diff2 = abs(b3['angle'] + 2*math.pi - b2['angle'])
                
                if angle_diff1 < self.MATCH_ANGLE_THRESHOLD and angle_diff2 < self.MATCH_ANGLE_THRESHOLD:
                    if i1 not in matched_indices or i2 not in matched_indices or i3 not in matched_indices:
                        num_matches += 1
                        matched_indices.update([i1, i2, i3])

        if matched_indices:
            self.score += len(matched_indices) * 10 # 10 points per block in a match
            
            # Create particle effects for matched blocks
            for i in sorted(list(matched_indices), reverse=True):
                block = self.belt_blocks[i]
                angle = block['angle'] + self.belt_angle
                x = self.BELT_CENTER[0] + self.BELT_RADIUS * math.cos(angle)
                y = self.BELT_CENTER[1] + self.BELT_RADIUS * math.sin(angle)
                self._create_explosion(x, y, block['color'])

            # Remove matched blocks
            self.belt_blocks = [b for i, b in enumerate(self.belt_blocks) if i not in matched_indices]
        
        return num_matches

    def _create_explosion(self, x, y, color):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'x': x, 'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': self.np_random.integers(20, 41),
                'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_belt()
        self._render_arms()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for i in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))
            
    def _render_belt(self):
        pygame.gfxdraw.filled_circle(self.screen, int(self.BELT_CENTER[0]), int(self.BELT_CENTER[1]), self.BELT_RADIUS, self.COLOR_BELT)
        pygame.gfxdraw.aacircle(self.screen, int(self.BELT_CENTER[0]), int(self.BELT_CENTER[1]), self.BELT_RADIUS, self.COLOR_BELT_RIM)
        pygame.gfxdraw.aacircle(self.screen, int(self.BELT_CENTER[0]), int(self.BELT_CENTER[1]), self.BELT_RADIUS - 20, self.COLOR_BELT_RIM)

        for block in self.belt_blocks:
            angle = block['angle'] + self.belt_angle
            x = self.BELT_CENTER[0] + self.BELT_RADIUS * math.cos(angle)
            y = self.BELT_CENTER[1] + self.BELT_RADIUS * math.sin(angle)
            self._draw_block(int(x), int(y), block['color'], block['glow'])

    def _render_arms(self):
        for i, arm in enumerate(self.arms):
            is_selected = (i == self.selected_arm_idx)
            arm_color = self.COLOR_ARM_SELECTED if is_selected else self.COLOR_ARM
            
            # Arm Base
            base_rect = pygame.Rect(arm['base_x'] - 15, self.ARM_Y_POS - 10, 30, 10)
            pygame.draw.rect(self.screen, self.COLOR_ARM_BASE, base_rect, border_radius=3)
            
            # Arm Body
            arm_rect = pygame.Rect(arm['x'] - self.ARM_WIDTH/2, self.ARM_Y_POS, self.ARM_WIDTH, self.ARM_HEIGHT)
            pygame.draw.rect(self.screen, arm_color, arm_rect, border_radius=3)
            
            if arm['held_block']:
                block_x = arm['x']
                block_y = self.ARM_Y_POS + self.ARM_HEIGHT + self.BLOCK_SIZE
                
                # Draw a small indicator if on cooldown
                if arm['spawn_timer'] > 0:
                    cooldown_progress = 1 - (arm['spawn_timer'] / self.BLOCK_SPAWN_DELAY)
                    pygame.draw.circle(self.screen, self.COLOR_ARM_SELECTED, (int(block_x), int(block_y)), int(self.BLOCK_SIZE * cooldown_progress), 1)
                else:
                    self._draw_block(int(block_x), int(block_y), arm['held_block']['color'], arm['held_block']['glow'])

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 40))
            color = p['color']
            s = self.screen.convert_alpha()
            s.fill((255, 255, 255, 0))
            pygame.draw.circle(s, (*color, alpha), (int(p['x']), int(p['y'])), int(p['life'] / 15 + 1))
            self.screen.blit(s, (0,0))
    
    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score:05d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Timer
        time_left = math.ceil(self.timer / self.FPS) if self.timer > 0 else 0
        timer_text = self.font_large.render(f"{time_left:02d}", True, self.COLOR_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(timer_text, timer_rect)

        # Game Over
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            game_over_text = self.font_large.render("TIME UP", True, self.COLOR_ARM_SELECTED)
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(game_over_text, text_rect)
            
            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            score_rect = final_score_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 20))
            self.screen.blit(final_score_text, score_rect)

    def _draw_block(self, x, y, color, glow_color):
        # Glow effect
        glow_radius = int(self.BLOCK_SIZE * 1.8)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*glow_color, 100), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (x - glow_radius, y - glow_radius))

        # Main block
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.BLOCK_SIZE, color)
        pygame.gfxdraw.aacircle(self.screen, x, y, self.BLOCK_SIZE, (255,255,255))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "selected_arm": self.selected_arm_idx
        }
        
    def close(self):
        pygame.quit()
        
    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_observation()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires pygame to be installed and will open a window.
    # Set the render_mode to None to run headless.
    env = GameEnv()
    obs, info = env.reset()
    
    # Pygame setup for manual play
    pygame.display.set_caption("Robo-Sorter")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Action defaults
        movement = 0  # none
        space_held = 0 # released
        shift_held = 0 # released

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Key handling
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(GameEnv.FPS)
        
        if terminated:
            # Pause for a moment before resetting
            pygame.time.wait(3000)
            obs, info = env.reset()

    env.close()