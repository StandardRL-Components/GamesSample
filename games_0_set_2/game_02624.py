
# Generated: 2025-08-27T20:56:20.750054
# Source Brief: brief_02624.md
# Brief Index: 2624

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑/↓ to move vertically. Avoid obstacles. Near misses give speed boosts if space is held."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro side-view racer. Dodge obstacles, complete 3 laps against the clock, and aim for the fastest time."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 10000

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_TRACK = (70, 70, 80)
    COLOR_TRACK_LINE = (100, 100, 110)
    COLOR_PLAYER = (255, 65, 54)
    COLOR_PLAYER_GLOW = (255, 120, 110)
    COLOR_OBSTACLE = (0, 31, 63)
    COLOR_OBSTACLE_GLOW = (60, 120, 180)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TIMER_WARN = (255, 220, 0)
    COLOR_TIMER_CRIT = (255, 65, 54)
    
    # Game parameters
    TRACK_Y_TOP = 80
    TRACK_Y_BOTTOM = 320
    TRACK_LANE_HEIGHT = 40
    TRACK_LENGTH = 5000  # Pixels per lap
    TOTAL_LAPS = 3
    TIME_LIMIT_SECONDS = 180
    
    # Player physics
    PLAYER_X_POS = 120
    PLAYER_WIDTH = 25
    PLAYER_HEIGHT = 15
    PLAYER_ACCEL = 1.0
    PLAYER_FRICTION = 0.85
    PLAYER_MAX_VEL = 8.0

    # Obstacle parameters
    OBSTACLE_MIN_SPEED = 4.0
    OBSTACLE_SPAWN_INTERVAL = 40
    OBSTACLE_SPEED_LAP_INCREASE = 1.5

    # Reward parameters
    REWARD_SURVIVAL = 0.01
    REWARD_NEAR_MISS = 0.5
    REWARD_SAFE_PLAY = -0.02
    REWARD_LAP_COMPLETE = 10.0
    REWARD_RACE_COMPLETE = 100.0
    REWARD_COLLISION = -50.0
    REWARD_TIMEOUT = -100.0
    NEAR_MISS_DISTANCE = 50
    SAFE_DISTANCE = 120

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_big = pygame.font.Font(None, 72)
        
        self.render_mode = render_mode
        self.np_random = None

        # Initialize state variables to avoid attribute errors
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.player_vel_y = 0.0
        self.obstacles = []
        self.particles = []
        self.lap_count = 0
        self.time_remaining = 0
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.base_scroll_speed = 0.0
        self.boost_timer = 0
        self.last_near_miss_frame = -100
        
        self.reset()
        # self.validate_implementation() # Uncomment to run validation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = [self.PLAYER_X_POS, self.HEIGHT / 2]
        self.player_vel_y = 0.0
        
        self.obstacles = []
        self.particles = []
        self.lap_count = 0
        
        self.time_remaining = self.TIME_LIMIT_SECONDS * self.FPS
        self.camera_x = 0.0
        self.camera_y = self.player_pos[1]
        
        self.base_scroll_speed = self.OBSTACLE_MIN_SPEED
        self.boost_timer = 0
        self.last_near_miss_frame = -100 # Cooldown for near miss reward

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Update game clock and state ---
        self.steps += 1
        self.time_remaining -= 1
        reward = 0
        
        # --- Handle player input ---
        self._handle_input(action)

        # --- Update game logic ---
        self._update_player()
        self._update_scroll_speed(action)
        self._update_camera()
        self._update_obstacles()
        self._update_particles()
        
        # --- Check for events and calculate rewards ---
        collision, terminated_by_collision = self._handle_collisions()
        if collision:
            reward = self.REWARD_COLLISION
            self.game_over = True
        else:
            # Continuous rewards only if no collision
            reward += self.REWARD_SURVIVAL
            
            # Near miss and safe play rewards
            closest_dist = self._get_closest_obstacle_dist()
            if closest_dist is not None:
                if closest_dist < self.NEAR_MISS_DISTANCE and self.steps > self.last_near_miss_frame + self.FPS:
                    reward += self.REWARD_NEAR_MISS
                    self.last_near_miss_frame = self.steps
                    # sfx: near_miss_whoosh.wav
                    if action[1] == 1: # Space held
                        self.boost_timer = 15 # frames
                        # sfx: boost_activate.wav
                elif closest_dist > self.SAFE_DISTANCE:
                    reward += self.REWARD_SAFE_PLAY

        # Lap completion
        if self.camera_x >= self.TRACK_LENGTH:
            self.camera_x -= self.TRACK_LENGTH
            self.lap_count += 1
            reward += self.REWARD_LAP_COMPLETE
            self.base_scroll_speed += self.OBSTACLE_SPEED_LAP_INCREASE
            # sfx: lap_complete.wav
            
            if self.lap_count >= self.TOTAL_LAPS:
                reward += self.REWARD_RACE_COMPLETE
                self.game_over = True

        # Termination conditions
        terminated = self.game_over
        if self.time_remaining <= 0 and not terminated:
            reward = self.REWARD_TIMEOUT
            terminated = True
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
        
        self.game_over = terminated
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        if movement == 1:  # Up
            self.player_vel_y -= self.PLAYER_ACCEL
        elif movement == 2:  # Down
            self.player_vel_y += self.PLAYER_ACCEL

    def _update_player(self):
        self.player_vel_y *= self.PLAYER_FRICTION
        self.player_vel_y = np.clip(self.player_vel_y, -self.PLAYER_MAX_VEL, self.PLAYER_MAX_VEL)
        self.player_pos[1] += self.player_vel_y
        
        # Clamp player to track boundaries
        self.player_pos[1] = np.clip(self.player_pos[1], 
                                     self.TRACK_Y_TOP + self.PLAYER_HEIGHT / 2, 
                                     self.TRACK_Y_BOTTOM - self.PLAYER_HEIGHT / 2)

    def _update_scroll_speed(self, action):
        current_scroll_speed = self.base_scroll_speed
        if self.boost_timer > 0:
            current_scroll_speed *= 2.0
            self.boost_timer -= 1
            # Add boost particles
            if self.steps % 2 == 0:
                p_x = self.player_pos[0] - self.PLAYER_WIDTH / 2
                p_y = self.player_pos[1]
                self.particles.append({
                    'pos': [p_x, p_y + self.np_random.uniform(-5, 5)],
                    'vel': [-self.np_random.uniform(2, 4), self.np_random.uniform(-1, 1)],
                    'life': 10, 'color': self.COLOR_TIMER_WARN
                })

        self.camera_x += current_scroll_speed

    def _update_camera(self):
        # Smoothly follow player's y position
        self.camera_y += (self.player_pos[1] - self.camera_y) * 0.1

    def _update_obstacles(self):
        # Spawn new obstacles
        last_obstacle_x = max([obs['pos'][0] for obs in self.obstacles] + [0])
        if self.WIDTH - last_obstacle_x > self.OBSTACLE_SPAWN_INTERVAL * (self.base_scroll_speed / self.OBSTACLE_MIN_SPEED):
            new_y = self.np_random.uniform(self.TRACK_Y_TOP, self.TRACK_Y_BOTTOM)
            new_width = self.np_random.integers(20, 50)
            new_height = self.np_random.integers(30, 70)
            self.obstacles.append({
                'pos': [self.WIDTH + new_width, new_y],
                'size': (new_width, new_height),
            })
        
        # Move and remove old obstacles
        scroll_speed = self.camera_x - (self.camera_x - self.base_scroll_speed) # A bit of a hack to get previous camera_x
        current_scroll_speed = self.base_scroll_speed * (2.0 if self.boost_timer > 0 else 1.0)
        
        for obs in self.obstacles:
            obs['pos'][0] -= current_scroll_speed
        
        self.obstacles = [obs for obs in self.obstacles if obs['pos'][0] > -obs['size'][0]]

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _handle_collisions(self):
        player_rect = pygame.Rect(
            self.player_pos[0] - self.PLAYER_WIDTH / 2,
            self.player_pos[1] - self.PLAYER_HEIGHT / 2,
            self.PLAYER_WIDTH, self.PLAYER_HEIGHT
        )
        for obs in self.obstacles:
            obs_rect = pygame.Rect(
                obs['pos'][0] - obs['size'][0] / 2,
                obs['pos'][1] - obs['size'][1] / 2,
                obs['size'][0], obs['size'][1]
            )
            if player_rect.colliderect(obs_rect):
                # sfx: explosion.wav
                self._create_explosion(self.player_pos)
                return True, True
        return False, False
        
    def _get_closest_obstacle_dist(self):
        player_center = np.array(self.player_pos)
        min_dist = float('inf')
        
        if not self.obstacles:
            return None
            
        for obs in self.obstacles:
            if obs['pos'][0] > self.player_pos[0]: # Only consider obstacles ahead
                obs_center = np.array(obs['pos'])
                dist = np.linalg.norm(player_center - obs_center)
                min_dist = min(min_dist, dist)
        
        return min_dist if min_dist != float('inf') else None

    def _create_explosion(self, pos):
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 8)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 30),
                'color': random.choice([self.COLOR_PLAYER, self.COLOR_TIMER_WARN, (255,255,255)])
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render track
        pygame.draw.rect(self.screen, self.COLOR_TRACK, (0, self.TRACK_Y_TOP, self.WIDTH, self.TRACK_Y_BOTTOM - self.TRACK_Y_TOP))
        for i in range(self.TRACK_Y_TOP, self.TRACK_Y_BOTTOM, self.TRACK_LANE_HEIGHT):
             line_x_offset = (self.camera_x * 0.5) % (self.TRACK_LANE_HEIGHT * 2)
             for x in range(int(-line_x_offset), self.WIDTH, self.TRACK_LANE_HEIGHT * 2):
                 pygame.draw.rect(self.screen, self.COLOR_TRACK_LINE, (x, i-2, self.TRACK_LANE_HEIGHT, 4))

        # Render finish line
        finish_x = self.TRACK_LENGTH - self.camera_x
        if finish_x < self.WIDTH + 50:
            for i in range(self.TRACK_Y_TOP, self.TRACK_Y_BOTTOM, 10):
                color = (255,255,255) if (i // 10) % 2 == 0 else (0,0,0)
                pygame.draw.rect(self.screen, color, (finish_x, i, 10, 10))
            for i in range(self.TRACK_Y_TOP, self.TRACK_Y_BOTTOM, 10):
                color = (0,0,0) if (i // 10) % 2 == 0 else (255,255,255)
                pygame.draw.rect(self.screen, color, (finish_x + 10, i, 10, 10))

        # Render obstacles
        for obs in self.obstacles:
            center_x, center_y = obs['pos']
            w, h = obs['size']
            rect = (center_x - w/2, center_y - h/2, w, h)
            # Glow effect
            glow_rect = (rect[0] - 2, rect[1] - 2, rect[2] + 4, rect[3] + 4)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_GLOW, glow_rect, border_radius=4)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect, border_radius=3)

        # Render particles
        for p in self.particles:
            size = max(1, p['life'] / 4)
            pygame.draw.rect(self.screen, p['color'], (p['pos'][0]-size/2, p['pos'][1]-size/2, size, size))

        # Render player
        if not (self.game_over and self.REWARD_COLLISION in [self.score, self.score - self.REWARD_LAP_COMPLETE]):
            px, py = int(self.player_pos[0]), int(self.player_pos[1])
            w, h = self.PLAYER_WIDTH, self.PLAYER_HEIGHT
            player_rect = (px - w/2, py - h/2, w, h)

            # Glow
            glow_surf = pygame.Surface((w + 10, h + 10), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*self.COLOR_PLAYER_GLOW, 100), glow_surf.get_rect(), border_radius=8)
            self.screen.blit(glow_surf, (px - w/2 - 5, py - h/2 - 5))
            
            # Car body
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
            # Windshield
            pygame.draw.rect(self.screen, (0,0,0), (px + 2, py - h/2 + 2, 6, h-4), border_radius=2)

    def _render_ui(self):
        # Time
        time_text = f"TIME: {max(0, self.time_remaining // self.FPS):03}"
        time_color = self.COLOR_TEXT
        if self.time_remaining < 10 * self.FPS: time_color = self.COLOR_TIMER_CRIT
        elif self.time_remaining < 30 * self.FPS: time_color = self.COLOR_TIMER_WARN
        self._draw_text(time_text, (self.WIDTH - 10, 10), self.font_ui, time_color, align="topright")
        
        # Laps
        lap_text = f"LAP: {min(self.lap_count + 1, self.TOTAL_LAPS)} / {self.TOTAL_LAPS}"
        self._draw_text(lap_text, (10, 10), self.font_ui, self.COLOR_TEXT, align="topleft")
        
        # Score
        score_text = f"SCORE: {int(self.score)}"
        self._draw_text(score_text, (self.WIDTH / 2, 10), self.font_ui, self.COLOR_TEXT, align="midtop")

        # Game Over message
        if self.game_over:
            if self.lap_count >= self.TOTAL_LAPS:
                msg = "RACE COMPLETE"
                color = (0, 255, 0)
            elif self.time_remaining <= 0:
                msg = "TIME UP"
                color = self.COLOR_TIMER_CRIT
            else: # Collision
                msg = "CRASHED"
                color = self.COLOR_PLAYER
            self._draw_text(msg, (self.WIDTH/2, self.HEIGHT/2), self.font_big, color, align="center")

    def _draw_text(self, text, pos, font, color, align="center"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "center":
            text_rect.center = pos
        elif align == "topleft":
            text_rect.topleft = pos
        elif align == "topright":
            text_rect.topright = pos
        elif align == "midtop":
            text_rect.midtop = pos
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lap": self.lap_count,
            "time_remaining": self.time_remaining // self.FPS
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Arcade Racer")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    running = True

    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
        if keys[pygame.K_r]: # Reset on 'r' key
            obs, info = env.reset()
        if keys[pygame.K_q] or keys[pygame.K_ESCAPE]:
             running = False

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render the observation to the display ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Laps: {info['lap']}, Steps: {info['steps']}")
            # Wait a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(GameEnv.FPS)

    env.close()