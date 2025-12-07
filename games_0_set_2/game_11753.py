import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:33:08.025817
# Source Brief: brief_01753.md
# Brief Index: 1753
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent pilots a rocket through a twisting neon tunnel.
    The goal is to hit all checkpoints and reach the end before time runs out, avoiding the walls.
    High-speed gameplay is rewarded through chaining checkpoints and using a temporary boost.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Pilot a rocket through a twisting neon tunnel, hitting all checkpoints to reach the "
        "end before time runs out while avoiding the walls."
    )
    user_guide = (
        "Controls: ↑ to accelerate, ↓ to decelerate, and ←→ to steer. "
        "Press space for a temporary speed boost."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 90.0
        self.MAX_STEPS = int(self.TIME_LIMIT_SECONDS * self.FPS)
        self.TOTAL_CHECKPOINTS = 20
        
        # --- Reward/Penalty Constants ---
        self.FINAL_CHECKPOINT_REWARD = 100.0
        self.WALL_HIT_PENALTY = -100.0
        self.TIME_OUT_PENALTY = -100.0
        self.CHECKPOINT_REWARD = 1.0
        self.CHAIN_BONUS_REWARD = 2.0
        self.SURVIVAL_REWARD = 0.01

        # --- Colors (Neon Theme) ---
        self.COLOR_BG = (10, 0, 20)
        self.COLOR_WALL = (40, 20, 80)
        self.COLOR_WALL_GLOW = (80, 60, 160)
        self.COLOR_ROCKET = (255, 50, 50)
        self.COLOR_ROCKET_GLOW = (255, 100, 100)
        self.COLOR_EXHAUST = (255, 200, 100)
        self.COLOR_CHECKPOINT = (50, 255, 50)
        self.COLOR_CHECKPOINT_GLOW = (150, 255, 150)
        self.COLOR_UI_TEXT = (220, 220, 255)
        self.COLOR_SPEED_LINE = (200, 200, 255, 100)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.Font(pygame.font.get_default_font(), 20)
            self.font_timer = pygame.font.Font(pygame.font.get_default_font(), 28)
        except IOError:
            self.font_ui = pygame.font.SysFont("monospace", 20)
            self.font_timer = pygame.font.SysFont("monospace", 28)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_remaining = 0.0
        self.player_pos = np.array([0.0, 0.0])
        self.player_vel_x = 0.0
        self.scroll_velocity = 0.0
        self.world_y = 0.0
        self.tunnel_segments = []
        self.checkpoints = []
        self.checkpoints_hit = 0
        self.checkpoint_chain = 0
        self.particles = []
        self.speed_lines = []
        self.boost_cooldown = 0

        # self.reset() is called by the environment wrapper, no need to call it here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Game State ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_remaining = self.TIME_LIMIT_SECONDS
        self.player_pos = np.array([self.WIDTH / 2.0, self.HEIGHT * 0.8])
        self.player_vel_x = 0.0
        self.scroll_velocity = 5.0
        self.world_y = 0.0
        self.checkpoints_hit = 0
        self.checkpoint_chain = 0
        self.particles.clear()
        self.speed_lines.clear()
        self.boost_cooldown = 0

        # --- Generate Tunnel & Checkpoints ---
        self.tunnel_segments.clear()
        self.checkpoints.clear()
        
        path_y = 0
        for i in range(self.TOTAL_CHECKPOINTS):
            # Generate tunnel segments leading to the next checkpoint
            num_segments_between = 60
            for _ in range(num_segments_between):
                path_y -= 15
                self._generate_next_segment(path_y, i)
            
            if self.tunnel_segments:
                last_segment = self.tunnel_segments[-1]
                self.checkpoints.append({
                    "pos": np.array([last_segment['center_x'], last_segment['y']]),
                    "active": True,
                    "radius": 20
                })
        
        # Add a final stretch of tunnel after the last checkpoint
        for _ in range(100):
            path_y -= 15
            self._generate_next_segment(path_y, self.TOTAL_CHECKPOINTS)
            
        return self._get_observation(), self._get_info()

    def _generate_next_segment(self, y_pos, checkpoints_passed):
        # Use sine waves for a smooth, winding path
        base_curve = math.sin(y_pos / 300.0) * (self.WIDTH / 4)
        detail_curve = math.sin(y_pos / 80.0) * (self.WIDTH / 10)
        center_x = self.WIDTH / 2.0 + base_curve + detail_curve
        
        # Difficulty scaling: tunnel narrows as more checkpoints are hit
        base_width = 180
        width_reduction = min(checkpoints_passed // 2 * 10, 80)
        width = base_width - width_reduction
        
        self.tunnel_segments.append({'y': y_pos, 'center_x': center_x, 'width': width})

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = self.SURVIVAL_REWARD

        # 1. Unpack and handle actions
        movement, space_held, _ = action
        
        player_accel_x = 0.0
        if movement == 3: player_accel_x = -1.2 # Steer Left
        elif movement == 4: player_accel_x = 1.2 # Steer Right
        
        if movement == 1: self.scroll_velocity = min(20.0, self.scroll_velocity + 0.2) # Accelerate
        elif movement == 2: self.scroll_velocity = max(3.0, self.scroll_velocity - 0.3) # Decelerate

        is_boosting = False
        if space_held and self.boost_cooldown == 0:
            self.scroll_velocity += 8.0
            self.boost_cooldown = 60 # 2 second cooldown
            is_boosting = True
            # Sound placeholder: Boost activate

        self.boost_cooldown = max(0, self.boost_cooldown - 1)

        # 2. Update Physics
        self.player_vel_x += player_accel_x
        self.player_vel_x *= 0.85  # Damping
        self.player_pos[0] += self.player_vel_x
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WIDTH)

        self.scroll_velocity = max(2.0, self.scroll_velocity * 0.995) # Natural speed decay
        self.world_y += self.scroll_velocity
        
        for seg in self.tunnel_segments: seg['y'] += self.scroll_velocity
        for cp in self.checkpoints: cp['pos'][1] += self.scroll_velocity

        # 3. Update Game Logic & Collisions
        # Wall Collision
        player_segment_idx = next((i for i, seg in enumerate(self.tunnel_segments) if seg['y'] > self.player_pos[1]), -1)
        
        if player_segment_idx > 0:
            p_seg, c_seg = self.tunnel_segments[player_segment_idx - 1], self.tunnel_segments[player_segment_idx]
            ratio = (self.player_pos[1] - p_seg['y']) / (c_seg['y'] - p_seg['y']) if (c_seg['y'] - p_seg['y']) != 0 else 0
            center_x = p_seg['center_x'] + (c_seg['center_x'] - p_seg['center_x']) * ratio
            width = p_seg['width'] + (c_seg['width'] - p_seg['width']) * ratio
            
            if not (center_x - width / 2 < self.player_pos[0] < center_x + width / 2):
                self.game_over = True
                reward = self.WALL_HIT_PENALTY
                # Sound placeholder: Crash

        # Checkpoint Collision
        hit_checkpoint_this_frame = False
        if not self.game_over:
            for cp in self.checkpoints:
                if cp['active'] and np.linalg.norm(self.player_pos - cp['pos']) < cp['radius'] + 15:
                    cp['active'] = False
                    self.checkpoints_hit += 1
                    self.checkpoint_chain += 1
                    hit_checkpoint_this_frame = True
                    reward += self.CHECKPOINT_REWARD
                    self.scroll_velocity = min(25.0, self.scroll_velocity + 3.0)
                    if self.checkpoint_chain >= 3:
                        reward += self.CHAIN_BONUS_REWARD
                        self.checkpoint_chain = 0
                        # Sound placeholder: Chain bonus
                    # Sound placeholder: Checkpoint hit
                    break
        
        if not hit_checkpoint_this_frame: self.checkpoint_chain = 0

        # 4. Update Effects
        self._update_particles(is_boosting)
        self._update_speed_lines(is_boosting or self.scroll_velocity > 18.0)

        # 5. Check Termination Conditions
        self.time_remaining -= 1.0 / self.FPS
        terminated = self.game_over
        truncated = False
        if not terminated:
            if self.checkpoints_hit >= self.TOTAL_CHECKPOINTS:
                terminated = True
                reward = self.FINAL_CHECKPOINT_REWARD
                # Sound placeholder: Victory
            elif self.time_remaining <= 0:
                terminated = True
                reward += self.TIME_OUT_PENALTY
            elif self.steps >= self.MAX_STEPS:
                truncated = True # Use truncated for time/step limit
        
        self.game_over = terminated or truncated
        self.score += reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_particles(self, is_boosting):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        
        for _ in range(2 + int(self.scroll_velocity / 4) * (3 if is_boosting else 1)):
            vel = np.array([self.player_vel_x * 0.1 + random.uniform(-0.5, 0.5), self.scroll_velocity * 0.1 + random.uniform(0, 1)])
            self.particles.append({'pos': self.player_pos.copy() + [random.uniform(-5, 5), 10], 'vel': vel, 'life': 30, 'max_life': 30, 'radius': random.uniform(1, 4)})

    def _update_speed_lines(self, is_fast):
        self.speed_lines = [sl for sl in self.speed_lines if sl['pos'][1] < self.HEIGHT]
        for sl in self.speed_lines: sl['pos'][1] += self.scroll_velocity * 1.5
        if is_fast and len(self.speed_lines) < 50:
            for _ in range(3):
                self.speed_lines.append({'pos': np.array([random.uniform(0, self.WIDTH), random.uniform(-50, 0)]), 'len': random.uniform(20, 80), 'width': random.randint(1, 2)})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for sl in self.speed_lines:
            start_pos = (int(sl['pos'][0]), int(sl['pos'][1]))
            end_pos = (int(sl['pos'][0]), int(sl['pos'][1] + sl['len']))
            pygame.draw.line(self.screen, self.COLOR_SPEED_LINE, start_pos, end_pos, sl['width'])

        visible_segments = [seg for seg in self.tunnel_segments if -100 < seg['y'] < self.HEIGHT + 100]
        for i in range(len(visible_segments) - 1):
            p1, p2 = visible_segments[i], visible_segments[i+1]
            l1, r1 = (p1['center_x'] - p1['width']/2, p1['y']), (p1['center_x'] + p1['width']/2, p1['y'])
            l2, r2 = (p2['center_x'] - p2['width']/2, p2['y']), (p2['center_x'] + p2['width']/2, p2['y'])
            
            points_left = [(int(l1[0]), int(l1[1])), (int(l2[0]), int(l2[1])), (int(l2[0]-20), int(l2[1])), (int(l1[0]-20), int(l1[1]))]
            points_right = [(int(r1[0]), int(r1[1])), (int(r2[0]), int(r2[1])), (int(r2[0]+20), int(r2[1])), (int(r1[0]+20), int(r1[1]))]
            
            pygame.gfxdraw.filled_polygon(self.screen, points_left, self.COLOR_WALL)
            pygame.gfxdraw.filled_polygon(self.screen, points_right, self.COLOR_WALL)

        for cp in self.checkpoints:
            if cp['active'] and 0 < cp['pos'][1] < self.HEIGHT:
                pos = (int(cp['pos'][0]), int(cp['pos'][1]))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(cp['radius'] * 1.5), (*self.COLOR_CHECKPOINT_GLOW, 80))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(cp['radius'] * 1.2), (*self.COLOR_CHECKPOINT_GLOW, 120))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], cp['radius'], self.COLOR_CHECKPOINT)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], cp['radius'], self.COLOR_CHECKPOINT)

        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), (*self.COLOR_EXHAUST, alpha))
        
        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        rocket_points = [(px, py - 15), (px - 10, py + 10), (px + 10, py + 10)]
        glow_points = [(px, py - 20), (px - 15, py + 15), (px + 15, py + 15)]
        
        pygame.gfxdraw.filled_polygon(self.screen, glow_points, (*self.COLOR_ROCKET_GLOW, 150))
        pygame.gfxdraw.aapolygon(self.screen, glow_points, (*self.COLOR_ROCKET_GLOW, 150))
        pygame.gfxdraw.filled_polygon(self.screen, rocket_points, self.COLOR_ROCKET)
        pygame.gfxdraw.aapolygon(self.screen, rocket_points, self.COLOR_ROCKET)

    def _render_ui(self):
        timer_surf = self.font_timer.render(f"{max(0, self.time_remaining):.1f}s", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_surf, (self.WIDTH - timer_surf.get_width() - 15, 10))

        speed_surf = self.font_ui.render(f"SPEED: {int(self.scroll_velocity * 10)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(speed_surf, (15, self.HEIGHT - 30))

        cp_surf = self.font_ui.render(f"CHECKPOINTS: {self.checkpoints_hit}/{self.TOTAL_CHECKPOINTS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(cp_surf, (self.WIDTH - cp_surf.get_width() - 15, self.HEIGHT - 30))
        
        score_surf = self.font_ui.render(f"SCORE: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (15, 15))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "time_remaining": self.time_remaining, "checkpoints_hit": self.checkpoints_hit, "speed": self.scroll_velocity}

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not run in a headless environment.
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tunnel Runner")
    clock = pygame.time.Clock()
    
    total_reward = 0.0

    while not done:
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        action = [movement, 1 if keys[pygame.K_SPACE] else 0, 1 if keys[pygame.K_LSHIFT] else 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0.0

        clock.tick(env.FPS)
        
    print(f"Game Over! Final Score: {total_reward:.2f}")
    env.close()