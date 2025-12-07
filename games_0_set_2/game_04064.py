
# Generated: 2025-08-28T01:18:18.977392
# Source Brief: brief_04064.md
# Brief Index: 4064

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑/↓ to steer. Hold SPACE for a temporary speed boost."
    )

    game_description = (
        "A minimalist, neon-infused side-scrolling racer. "
        "Dodge obstacles, hit checkpoints, and complete three laps as fast as possible."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30 # Assumed frame rate for physics
        self.MAX_STEPS = 3000 # Max episode length (3000 steps = 100 seconds at 30fps)

        # Visuals
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (255, 50, 50)
        self.COLOR_PLAYER_GLOW = (255, 50, 50)
        self.COLOR_OBSTACLE = (50, 150, 255)
        self.COLOR_OBSTACLE_GLOW = (50, 150, 255)
        self.COLOR_CHECKPOINT = (50, 255, 150)
        self.COLOR_TRACK = (220, 220, 220)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_UI_SHADOW = (0, 0, 0)
        self.COLOR_UI_BG = (0, 0, 0, 128)
        self.COLOR_BOOST_READY = (50, 255, 150)
        self.COLOR_BOOST_COOLDOWN = (255, 50, 50)

        # Gameplay
        self.TRACK_LENGTH = 3000
        self.NUM_CHECKPOINTS_PER_LAP = 4
        self.PLAYER_SIZE = (25, 12)
        self.OBSTACLE_RADIUS = 15
        self.PLAYER_X_ON_SCREEN = 100
        self.TRACK_Y_TOP = 60
        self.TRACK_Y_BOTTOM = self.HEIGHT - 60
        self.PLAYER_V_SPEED = 4.0
        self.BOOST_DURATION = 15 # 0.5s at 30fps
        self.BOOST_COOLDOWN_TOTAL = 90 # 3s at 30fps
        
        # --- Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        # --- State Variables ---
        # These are initialized here to None/empty and properly set in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.player_pos = [0, 0]
        self.base_speed = 0
        self.current_speed = 0
        
        self.lap = 1
        self.lap_start_step = 0
        self.lap_times = []
        
        self.obstacles = []
        self.checkpoints = []
        self.passed_checkpoints = set()

        self.boost_timer = 0
        self.boost_cooldown = 0
        
        self.particles = []
        self.camera_x = 0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_pos = [self.PLAYER_X_ON_SCREEN, self.HEIGHT / 2]
        self.lap = 1
        self.lap_start_step = 0
        self.lap_times = []
        
        self.boost_timer = 0
        self.boost_cooldown = 0
        
        self.particles = []
        
        self._generate_lap_assets()

        return self._get_observation(), self._get_info()

    def _generate_lap_assets(self):
        self.obstacles = []
        self.checkpoints = []
        self.passed_checkpoints = set()
        self.player_pos[0] = self.PLAYER_X_ON_SCREEN # Reset horizontal world position

        num_obstacles = 3 + (self.lap - 1) * 2
        self.base_speed = 5.0 + (self.lap - 1) * 0.5
        self.current_speed = self.base_speed

        # Generate checkpoints
        for i in range(1, self.NUM_CHECKPOINTS_PER_LAP + 1):
            x = (i / (self.NUM_CHECKPOINTS_PER_LAP + 1)) * self.TRACK_LENGTH
            self.checkpoints.append(pygame.Rect(x, self.TRACK_Y_TOP, 10, self.TRACK_Y_BOTTOM - self.TRACK_Y_TOP))

        # Generate obstacles
        for _ in range(num_obstacles):
            while True:
                x = self.np_random.integers(300, self.TRACK_LENGTH - 200)
                y = self.np_random.integers(self.TRACK_Y_TOP + self.OBSTACLE_RADIUS, self.TRACK_Y_BOTTOM - self.OBSTACLE_RADIUS)
                
                # Ensure it's not too close to a checkpoint
                is_too_close_to_checkpoint = any(abs(x - cp.x) < 100 for cp in self.checkpoints)
                # Ensure it's not too close to another obstacle
                is_too_close_to_obstacle = any(math.hypot(x - obs_x, y - obs_y) < 100 for obs_x, obs_y, _ in self.obstacles)

                if not is_too_close_to_checkpoint and not is_too_close_to_obstacle:
                    self.obstacles.append((x, y, self.OBSTACLE_RADIUS))
                    break

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.1 # Survival reward
        terminated = False
        
        # --- Action Handling ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        if movement == 1: # Up
            self.player_pos[1] -= self.PLAYER_V_SPEED
        elif movement == 2: # Down
            self.player_pos[1] += self.PLAYER_V_SPEED
        elif movement == 0: # No-op penalty
            reward -= 0.2

        self.player_pos[1] = np.clip(self.player_pos[1], self.TRACK_Y_TOP + self.PLAYER_SIZE[1]/2, self.TRACK_Y_BOTTOM - self.PLAYER_SIZE[1]/2)
        
        # Boost
        if space_held and self.boost_cooldown == 0:
            self.boost_timer = self.BOOST_DURATION
            self.boost_cooldown = self.BOOST_COOLDOWN_TOTAL
            # Sound: Boost activate

        # --- Physics & Game Logic Update ---
        self.steps += 1
        
        if self.boost_timer > 0:
            self.boost_timer -= 1
            self.current_speed = self.base_speed * 1.5
            if self.boost_timer == 0:
                # Sound: Boost end
                pass
        else:
            self.current_speed = self.base_speed
            
        if self.boost_cooldown > 0:
            self.boost_cooldown -= 1
            
        self.player_pos[0] += self.current_speed
        
        player_rect = pygame.Rect(self.PLAYER_X_ON_SCREEN - self.PLAYER_SIZE[0]/2, self.player_pos[1] - self.PLAYER_SIZE[1]/2, *self.PLAYER_SIZE)
        
        # --- Collision Detection ---
        # Obstacles
        for ox, oy, orad in self.obstacles:
            screen_ox = ox - self.camera_x
            if abs(screen_ox - self.PLAYER_X_ON_SCREEN) < self.OBSTACLE_RADIUS + self.PLAYER_SIZE[0]: # Broad-phase
                if self._check_rect_circle_collision(player_rect, (screen_ox, oy), orad):
                    self.game_over = True
                    terminated = True
                    reward -= 50
                    # Sound: Crash
                    break
        if terminated:
            self.score += reward
            return self._get_observation(), reward, terminated, False, self._get_info()

        # Checkpoints
        for i, cp_rect in enumerate(self.checkpoints):
            if i not in self.passed_checkpoints:
                screen_cp_x = cp_rect.x - self.camera_x
                if player_rect.x > screen_cp_x:
                    self.passed_checkpoints.add(i)
                    reward += 1.0
                    # Sound: Checkpoint
        
        # Lap Completion
        if self.player_pos[0] >= self.TRACK_LENGTH:
            reward += 5.0
            self.lap += 1
            
            lap_duration = (self.steps - self.lap_start_step) / self.FPS
            self.lap_times.append(lap_duration)
            self.lap_start_step = self.steps
            
            if self.lap > 3:
                self.win = True
                self.game_over = True
                terminated = True
                reward += 50
                # Sound: Win jingle
            else:
                self._generate_lap_assets()
                # Sound: Lap complete

        # Max steps termination
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _check_rect_circle_collision(self, rect, circle_center, circle_radius):
        closest_x = max(rect.left, min(circle_center[0], rect.right))
        closest_y = max(rect.top, min(circle_center[1], rect.bottom))
        distance_sq = (closest_x - circle_center[0])**2 + (closest_y - circle_center[1])**2
        return distance_sq < (circle_radius**2)

    def _get_observation(self):
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self.screen.fill(self.COLOR_BG)
        self.camera_x = self.player_pos[0] - self.PLAYER_X_ON_SCREEN

        # --- Update & Draw Particles (Speed Lines) ---
        # Add new particles
        if self.current_speed > 0:
            num_new_particles = 1 + int(self.current_speed / self.base_speed)
            for _ in range(num_new_particles):
                p_y = self.player_pos[1] + self.np_random.uniform(-self.PLAYER_SIZE[1]/2, self.PLAYER_SIZE[1]/2)
                p_life = self.np_random.integers(10, 20)
                p_len = self.np_random.uniform(5, 15)
                self.particles.append([self.PLAYER_X_ON_SCREEN, p_y, p_life, p_len])
        
        # Update and draw existing particles
        remaining_particles = []
        for p in self.particles:
            p[0] -= self.current_speed * 1.5 # Move faster than camera for trail effect
            p[2] -= 1
            if p[2] > 0:
                remaining_particles.append(p)
                alpha = int(255 * (p[2] / 20))
                color = (*self.COLOR_TRACK, alpha)
                start_pos = (int(p[0]), int(p[1]))
                end_pos = (int(p[0] + p[3]), int(p[1]))
                pygame.draw.line(self.screen, color, start_pos, end_pos, 1)
        self.particles = remaining_particles

        # --- Draw Track ---
        pygame.draw.line(self.screen, self.COLOR_TRACK, (0, self.TRACK_Y_TOP), (self.WIDTH, self.TRACK_Y_TOP), 2)
        pygame.draw.line(self.screen, self.COLOR_TRACK, (0, self.TRACK_Y_BOTTOM), (self.WIDTH, self.TRACK_Y_BOTTOM), 2)

        # --- Draw Checkpoints ---
        for i, cp_rect in enumerate(self.checkpoints):
            screen_x = cp_rect.x - self.camera_x
            if 0 < screen_x < self.WIDTH:
                color = self.COLOR_CHECKPOINT if i not in self.passed_checkpoints else self.COLOR_TRACK
                alpha_surface = pygame.Surface(cp_rect.size, pygame.SRCALPHA)
                alpha_surface.fill((*color, 60))
                self.screen.blit(alpha_surface, (screen_x, cp_rect.y))
                pygame.draw.rect(self.screen, color, (screen_x, cp_rect.y, cp_rect.width, cp_rect.height), 2)

        # --- Draw Obstacles ---
        for x, y, rad in self.obstacles:
            screen_x = x - self.camera_x
            if 0 < screen_x < self.WIDTH:
                # Glow effect
                glow_rad = int(rad * 1.5)
                s = pygame.Surface((glow_rad * 2, glow_rad * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*self.COLOR_OBSTACLE_GLOW, 30), (glow_rad, glow_rad), glow_rad)
                self.screen.blit(s, (int(screen_x - glow_rad), int(y - glow_rad)))
                
                # Main circle
                pygame.gfxdraw.filled_circle(self.screen, int(screen_x), int(y), rad, self.COLOR_OBSTACLE)
                pygame.gfxdraw.aacircle(self.screen, int(screen_x), int(y), rad, self.COLOR_OBSTACLE)

        # --- Draw Player ---
        player_rect = pygame.Rect(0, 0, *self.PLAYER_SIZE)
        player_rect.center = (self.PLAYER_X_ON_SCREEN, self.player_pos[1])
        
        # Glow / Boost effect
        glow_size = int(self.PLAYER_SIZE[0] * (2.0 + (self.boost_timer > 0) * 1.5))
        s = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        glow_alpha = 50 + (self.boost_timer > 0) * 50
        pygame.draw.ellipse(s, (*self.COLOR_PLAYER_GLOW, glow_alpha), s.get_rect())
        self.screen.blit(s, (player_rect.centerx - glow_size//2, player_rect.centery - glow_size//2))

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos, shadow_color, shadow_offset=(2,2)):
            text_surf = font.render(text, True, shadow_color)
            self.screen.blit(text_surf, (pos[0] + shadow_offset[0], pos[1] + shadow_offset[1]))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        # Lap counter
        draw_text(f"LAP: {self.lap}/3", self.font_small, self.COLOR_UI_TEXT, (10, 10), self.COLOR_UI_SHADOW)
        
        # Checkpoints
        cp_text = f"CP: {len(self.passed_checkpoints)}/{self.NUM_CHECKPOINTS_PER_LAP}"
        draw_text(cp_text, self.font_small, self.COLOR_UI_TEXT, (10, 35), self.COLOR_UI_SHADOW)

        # Time
        time_text = f"TIME: {self.steps / self.FPS:.2f}"
        draw_text(time_text, self.font_small, self.COLOR_UI_TEXT, (self.WIDTH - 150, 10), self.COLOR_UI_SHADOW)
        
        # Lap times
        for i, t in enumerate(self.lap_times):
            lap_time_text = f"L{i+1}: {t:.2f}s"
            draw_text(lap_time_text, self.font_small, self.COLOR_UI_TEXT, (self.WIDTH - 150, 35 + i*20), self.COLOR_UI_SHADOW)

        # Boost Meter
        boost_bar_bg = pygame.Rect(10, self.HEIGHT - 30, 100, 20)
        pygame.draw.rect(self.screen, self.COLOR_UI_SHADOW, boost_bar_bg, border_radius=3)
        if self.boost_cooldown == 0:
            fill_ratio = 1.0
            bar_color = self.COLOR_BOOST_READY
            boost_text = "BOOST READY"
        else:
            fill_ratio = 1 - (self.boost_cooldown / self.BOOST_COOLDOWN_TOTAL)
            bar_color = self.COLOR_BOOST_COOLDOWN
            boost_text = "COOLDOWN"
        boost_bar_fill = pygame.Rect(10, self.HEIGHT - 30, 100 * fill_ratio, 20)
        pygame.draw.rect(self.screen, bar_color, boost_bar_fill, border_radius=3)
        draw_text(boost_text, self.font_small, self.COLOR_UI_TEXT, (120, self.HEIGHT - 29), self.COLOR_UI_SHADOW)

        # Game Over / Win Text
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,150))
            self.screen.blit(s, (0,0))
            message = "YOU WIN!" if self.win else "GAME OVER"
            text_surf = self.font_large.render(message, True, self.COLOR_UI_TEXT)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lap": self.lap,
            "lap_times": self.lap_times,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Arcade Racer")
    
    terminated = False
    total_reward = 0
    
    # --- Control Mapping ---
    # This maps keyboard keys to the MultiDiscrete action space
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3, # No effect in this game
        pygame.K_RIGHT: 4, # No effect in this game
    }

    action = np.array([0, 0, 0]) # [movement, space, shift]

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        # Get key presses
        keys = pygame.key.get_pressed()
        
        # Movement (mutually exclusive)
        action[0] = 0 # Default to no-op
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        
        # Space and Shift (can be simultaneous)
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        # Step the environment
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        terminated = term
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # Print info
        if env.steps % 30 == 0:
            print(f"Step: {info['steps']}, Lap: {info['lap']}, Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")

        env.clock.tick(env.FPS)

    print("Game Over!")
    print(f"Final Score: {info['score']:.2f}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Lap Times: {[f'{t:.2f}s' for t in info['lap_times']]}")

    env.close()