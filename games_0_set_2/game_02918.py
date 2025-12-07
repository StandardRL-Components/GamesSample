
# Generated: 2025-08-27T21:48:59.190943
# Source Brief: brief_02918.md
# Brief Index: 2918

        
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
        "Controls: Use ←→ to steer. Hold SPACE to accelerate, SHIFT to decelerate. Press ↑ to jump over obstacles."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, retro-futuristic racer. Navigate a treacherous, procedurally generated track and reach the finish line before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = self.FPS * 60  # 60-second time limit
        self.TRACK_LENGTH = 10000

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
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
        self.COLOR_BG = (10, 0, 20)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (0, 255, 128, 50)
        self.COLOR_OBSTACLE = (255, 50, 50)
        self.COLOR_OBSTACLE_GLOW = (255, 50, 50, 50)
        self.COLOR_TRACK = (0, 128, 255)
        self.COLOR_FINISH = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 240)

        # Physics constants
        self.GRAVITY = 0.8
        self.JUMP_STRENGTH = 14
        self.ACCELERATION = 0.4
        self.DECELERATION = 0.8
        self.MAX_SPEED = 15
        self.FRICTION = 0.98
        self.STEER_SPEED = 5
        self.GROUND_Y = self.HEIGHT - 80

        # Will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.robot_pos = None
        self.robot_vel = None
        self.is_on_ground = False
        self.world_progress = 0
        self.obstacles = []
        self.last_obstacle_spawn_pos = 0
        self.particles = []
        self.stars = []
        self.camera_shake = 0

        # Initialize state variables
        self.reset()
        
        # Run validation check
        # self.validate_implementation() # Commented out for final submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.robot_pos = [self.WIDTH // 4, self.GROUND_Y]
        self.robot_vel = [0, 0]
        self.is_on_ground = True
        self.world_progress = 0
        
        self.obstacles = []
        self.last_obstacle_spawn_pos = 0
        self.particles = []
        self.camera_shake = 0
        
        self._generate_stars()
        self._generate_obstacles(self.WIDTH * 2) # Pre-generate some obstacles

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        accelerate_held = action[1] == 1
        decelerate_held = action[2] == 1
        
        # --- Handle player input ---
        if accelerate_held:
            self.robot_vel[0] = min(self.MAX_SPEED, self.robot_vel[0] + self.ACCELERATION)
            if self.is_on_ground: # Thruster particles
                self._create_particles(
                    count=2, pos=[self.robot_pos[0] - 15, self.robot_pos[1] - 10], 
                    vel_range=([-10, -5], [-2, 2]), life_range=(5, 15), 
                    color=(255, 200, 50)
                )

        if decelerate_held:
            self.robot_vel[0] = max(0, self.robot_vel[0] - self.DECELERATION)

        if movement == 1 and self.is_on_ground: # Jump
            self.robot_vel[1] = -self.JUMP_STRENGTH
            self.is_on_ground = False
            # sfx: jump_sound()
            self._create_particles(
                count=10, pos=[self.robot_pos[0], self.robot_pos[1]],
                vel_range=([-3, 3], [-5, 0]), life_range=(10, 20),
                color=self.COLOR_TRACK
            )
            
        if movement == 3: # Steer left (up screen)
            self.robot_pos[1] = max(self.GROUND_Y - 100, self.robot_pos[1] - self.STEER_SPEED)
        if movement == 4: # Steer right (down screen)
            self.robot_pos[1] = min(self.GROUND_Y, self.robot_pos[1] + self.STEER_SPEED)
        
        # --- Update physics and game state ---
        self.robot_vel[0] *= self.FRICTION
        self.world_progress += self.robot_vel[0]
        
        self.robot_vel[1] += self.GRAVITY
        self.robot_pos[1] += self.robot_vel[1]
        
        if self.robot_pos[1] >= self.GROUND_Y:
            self.robot_pos[1] = self.GROUND_Y
            if not self.is_on_ground: # Landing
                self.camera_shake = 5
                # sfx: land_sound()
            self.is_on_ground = True
            self.robot_vel[1] = 0

        self._update_particles()
        if self.camera_shake > 0:
            self.camera_shake -= 1

        self._generate_obstacles(self.world_progress + self.WIDTH)
        self._prune_obstacles()
        
        self.steps += 1
        
        # --- Check for termination and calculate reward ---
        reward = 0
        terminated = False
        
        player_rect = pygame.Rect(self.robot_pos[0] - 15, self.robot_pos[1] - 30, 30, 30)

        # Collision with obstacles
        for obs in self.obstacles:
            obs_screen_x = obs['pos'][0] - self.world_progress
            obs_rect = pygame.Rect(obs_screen_x, obs['pos'][1] - obs['size'], obs['size'], obs['size'])
            if player_rect.colliderect(obs_rect):
                terminated = True
                reward = -10
                self.game_over = True
                self.camera_shake = 20
                # sfx: explosion_sound()
                self._create_particles(
                    count=50, pos=player_rect.center, 
                    vel_range=([-8, 8], [-8, 8]), life_range=(20, 40), 
                    color=self.COLOR_OBSTACLE
                )
                break
        
        # Reward for forward movement
        if not terminated:
            reward += self.robot_vel[0] * 0.01

        # Reaching the finish line
        if not terminated and self.world_progress >= self.TRACK_LENGTH:
            terminated = True
            time_bonus = 100 * (self.MAX_STEPS - self.steps) / self.MAX_STEPS
            reward = 50 + time_bonus
            self.game_over = True
            # sfx: win_sound()
        
        # Time ran out
        if not terminated and self.steps >= self.MAX_STEPS:
            terminated = True
            # sfx: lose_sound()

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        render_offset = [0, 0]
        if self.camera_shake > 0:
            render_offset[0] = self.np_random.integers(-self.camera_shake, self.camera_shake)
            render_offset[1] = self.np_random.integers(-self.camera_shake, self.camera_shake)

        self.screen.fill(self.COLOR_BG)
        
        self._draw_stars(render_offset)
        self._draw_track(render_offset)
        self._draw_finish_line(render_offset)
        self._draw_obstacles(render_offset)
        if not (self.game_over and self.score <= -10): # Don't draw player if they crashed
            self._draw_robot(render_offset)
        self._draw_particles(render_offset)
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "progress": self.world_progress / self.TRACK_LENGTH,
            "speed": self.robot_vel[0]
        }

    def _render_ui(self):
        # Timer
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        timer_text = f"TIME: {time_left:.1f}"
        text_surface = self.font_small.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (self.WIDTH - text_surface.get_width() - 10, 10))

        # Speed
        speed_text = f"SPEED: {self.robot_vel[0]:.1f}"
        text_surface = self.font_small.render(speed_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, self.HEIGHT - text_surface.get_height() - 10))
        
        # Progress Bar
        progress_ratio = self.world_progress / self.TRACK_LENGTH
        bar_width = self.WIDTH - 20
        bar_height = 5
        pygame.draw.rect(self.screen, (50,50,80), (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (10, 10, max(0, bar_width * progress_ratio), bar_height))

        if self.game_over:
            if self.world_progress >= self.TRACK_LENGTH:
                end_text = "FINISH!"
                end_color = self.COLOR_FINISH
            else:
                end_text = "GAME OVER"
                end_color = self.COLOR_OBSTACLE
            text_surface = self.font_large.render(end_text, True, end_color)
            pos = (self.WIDTH // 2 - text_surface.get_width() // 2, self.HEIGHT // 2 - text_surface.get_height() // 2)
            self.screen.blit(text_surface, pos)

    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'pos': [self.np_random.random() * self.WIDTH, self.np_random.random() * self.HEIGHT],
                'speed': 0.2 + self.np_random.random() * 0.8,
                'size': int(1 + self.np_random.random() * 2)
            })

    def _draw_stars(self, offset):
        for star in self.stars:
            star['pos'][0] -= star['speed'] * self.robot_vel[0]
            if star['pos'][0] < 0:
                star['pos'][0] = self.WIDTH
                star['pos'][1] = self.np_random.random() * self.HEIGHT
            
            pos = (int(star['pos'][0] + offset[0]), int(star['pos'][1] + offset[1]))
            color_val = int(100 * star['speed'])
            pygame.draw.circle(self.screen, (color_val, color_val, color_val + 50), pos, star['size'])

    def _draw_track(self, offset):
        # Ground line
        pygame.gfxdraw.hline(self.screen, 0, self.WIDTH, int(self.GROUND_Y + offset[1]), self.COLOR_TRACK)
        pygame.gfxdraw.hline(self.screen, 0, self.WIDTH, int(self.GROUND_Y + 1 + offset[1]), self.COLOR_TRACK)
        # Top boundary-ish line for parallax effect
        pygame.gfxdraw.hline(self.screen, 0, self.WIDTH, int(self.GROUND_Y - 120 + offset[1]), (20,40,80))

    def _draw_finish_line(self, offset):
        finish_x = self.TRACK_LENGTH - self.world_progress + offset[0]
        if 0 < finish_x < self.WIDTH:
            for i in range(0, self.HEIGHT, 10):
                color = self.COLOR_FINISH if (i // 10) % 2 == 0 else (0,0,0)
                pygame.draw.line(self.screen, color, (finish_x, i), (finish_x, i + 10), 3)

    def _generate_obstacles(self, until_pos):
        spawn_pos = self.last_obstacle_spawn_pos
        while spawn_pos < until_pos:
            spawn_pos += 150 + self.np_random.random() * 200
            if spawn_pos < 500: # Safe zone at start
                continue

            density = 0.5 + (self.steps / self.MAX_STEPS) * 0.4 # Density increases over time
            if self.np_random.random() < density:
                shape_type = self.np_random.choice(['square', 'triangle', 'circle'])
                size = self.np_random.integers(20, 40)
                y_pos = self.GROUND_Y - self.np_random.integers(0, 80)
                self.obstacles.append({'pos': [spawn_pos, y_pos], 'size': size, 'shape': shape_type})
        self.last_obstacle_spawn_pos = spawn_pos
        
    def _prune_obstacles(self):
        self.obstacles = [obs for obs in self.obstacles if obs['pos'][0] > self.world_progress - 100]

    def _draw_obstacles(self, offset):
        for obs in self.obstacles:
            screen_x = int(obs['pos'][0] - self.world_progress + offset[0])
            screen_y = int(obs['pos'][1] + offset[1])
            size = obs['size']

            if -size < screen_x < self.WIDTH + size:
                if obs['shape'] == 'square':
                    rect = pygame.Rect(screen_x, screen_y - size, size, size)
                    pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect, border_radius=3)
                elif obs['shape'] == 'triangle':
                    points = [(screen_x, screen_y), (screen_x + size, screen_y), (screen_x + size/2, screen_y - size)]
                    pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_OBSTACLE)
                    pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_OBSTACLE)
                elif obs['shape'] == 'circle':
                    pygame.gfxdraw.aacircle(self.screen, screen_x + size//2, screen_y - size//2, size//2, self.COLOR_OBSTACLE)
                    pygame.gfxdraw.filled_circle(self.screen, screen_x + size//2, screen_y - size//2, size//2, self.COLOR_OBSTACLE)

    def _draw_robot(self, offset):
        x, y = int(self.robot_pos[0] + offset[0]), int(self.robot_pos[1] + offset[1])
        w, h = 30, 30
        
        # Squash and stretch for jumping/landing
        squash = max(0, self.robot_vel[1] * 0.5)
        stretch = max(0, -self.robot_vel[1] * 0.5)
        
        rect = pygame.Rect(x - w/2, y - h - squash, w, h + squash - stretch)
        
        # Glow effect
        glow_surf = pygame.Surface((w * 2, (h + squash - stretch) * 2), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PLAYER_GLOW, (w/2, (h+squash-stretch)/2, w, h+squash-stretch), border_radius=10)
        self.screen.blit(glow_surf, (rect.x - w/2, rect.y - (h+squash-stretch)/2))
        
        # Main body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect, border_radius=5)
        
        # Eye
        eye_pos = (rect.centerx + 5, rect.centery - 5)
        pygame.draw.circle(self.screen, (255, 255, 255), eye_pos, 3)

    def _create_particles(self, count, pos, vel_range, life_range, color):
        for _ in range(count):
            self.particles.append({
                'pos': list(pos),
                'vel': [self.np_random.uniform(vel_range[0][0], vel_range[0][1]),
                        self.np_random.uniform(vel_range[1][0], vel_range[1][1])],
                'life': self.np_random.integers(life_range[0], life_range[1]),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _draw_particles(self, offset):
        for p in self.particles:
            screen_x = int(p['pos'][0] - self.world_progress + offset[0])
            screen_y = int(p['pos'][1] + offset[1])
            size = max(0, int(p['life'] * 0.2))
            pygame.draw.circle(self.screen, p['color'], (screen_x, screen_y), size)

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # env.validate_implementation() # Run the validator
    
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Robo-Racer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    action = env.action_space.sample()
    action = [0, 0, 0] # Start with no-op
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        
        # Map keys to MultiDiscrete action
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        # elif keys[pygame.K_DOWN]: movement = 2 # Unused
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    env.close()