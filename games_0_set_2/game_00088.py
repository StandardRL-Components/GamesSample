
# Generated: 2025-08-27T12:33:59.173452
# Source Brief: brief_00088.md
# Brief Index: 88

        
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
        "Controls: Press space to jump over the obstacles in time with the beat."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro-futuristic rhythm runner. Jump over neon obstacles to the beat. "
        "Perfect timing earns more points. Miss three beats and you're out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GROUND_Y = 350
        self.MAX_STEPS = 5000
        self.FINISH_DISTANCE = 4800 # in steps

        # Player constants
        self.PLAYER_X = 100
        self.PLAYER_WIDTH = 30
        self.PLAYER_HEIGHT = 40
        self.GRAVITY = 1.2
        self.JUMP_STRENGTH = -18

        # Obstacle constants
        self.OBSTACLE_WIDTH = 25
        self.OBSTACLE_HEIGHT = 35
        self.INITIAL_OBSTACLE_SPEED = 6.0
        self.INITIAL_OBSTACLE_INTERVAL = 90 # steps between spawns

        # Rhythm constants
        self.BEAT_PERIOD = 30 # A beat every 30 frames (1 per second at 30fps)
        self.PERFECT_WINDOW = 3 # frames on either side of the beat

        # Colors
        self.COLOR_BG_TOP = (15, 10, 40)
        self.COLOR_BG_BOTTOM = (40, 20, 70)
        self.COLOR_GROUND = (100, 80, 150)
        self.COLOR_PLAYER = (255, 0, 150)
        self.COLOR_PLAYER_GLOW = (255, 100, 200, 50)
        self.COLOR_OBSTACLE = (0, 255, 255)
        self.COLOR_OBSTACLE_GLOW = (100, 255, 255, 50)
        self.COLOR_BEAT_INDICATOR = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 255)
        self.COLOR_PARTICLE_GOOD = (255, 215, 0)
        self.COLOR_PARTICLE_BAD = (255, 50, 50)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)

        # Initialize state variables
        self.player_pos = None
        self.player_vel_y = None
        self.is_jumping = None
        self.obstacles = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.missed_beats = None
        self.obstacle_speed = None
        self.obstacle_interval = None
        self.next_obstacle_spawn_step = None
        self.np_random = None

        self.reset()
        
        # self.validate_implementation() # Optional: Call for debugging

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.player_pos = [self.PLAYER_X, self.GROUND_Y - self.PLAYER_HEIGHT]
        self.player_vel_y = 0
        self.is_jumping = False
        self.obstacles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.missed_beats = 0
        self.obstacle_speed = self.INITIAL_OBSTACLE_SPEED
        self.obstacle_interval = self.INITIAL_OBSTACLE_INTERVAL
        self.next_obstacle_spawn_step = 60
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1

        # Unpack factorized action
        space_pressed = action[1] == 1

        self._update_player(space_pressed)
        reward += self._update_obstacles()
        self._update_particles()
        self._spawn_obstacles()
        self._update_difficulty()

        if self.is_jumping:
            reward -= 0.1

        terminated = self._check_termination()
        if terminated:
            if self.steps >= self.FINISH_DISTANCE:
                reward += 100 # Reached finish line
                # sfx: victory_sound
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, space_pressed):
        if space_pressed and not self.is_jumping:
            self.player_vel_y = self.JUMP_STRENGTH
            self.is_jumping = True
            # sfx: jump_sound

        self.player_vel_y += self.GRAVITY
        self.player_pos[1] += self.player_vel_y

        if self.player_pos[1] >= self.GROUND_Y - self.PLAYER_HEIGHT:
            self.player_pos[1] = self.GROUND_Y - self.PLAYER_HEIGHT
            self.player_vel_y = 0
            if self.is_jumping:
                # sfx: land_sound
                self._create_particles([self.player_pos[0] + self.PLAYER_WIDTH/2, self.GROUND_Y], self.COLOR_PLAYER, 5, is_landing=True)
            self.is_jumping = False

    def _update_obstacles(self):
        step_reward = 0
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_WIDTH, self.PLAYER_HEIGHT)

        for obs in reversed(self.obstacles):
            obs['x'] -= self.obstacle_speed
            obs_rect = pygame.Rect(obs['x'], obs['y'], self.OBSTACLE_WIDTH, self.OBSTACLE_HEIGHT)

            if player_rect.colliderect(obs_rect):
                step_reward -= 5  # Penalty for collision
                self.missed_beats += 1
                self.obstacles.remove(obs)
                self._create_particles([obs['x'], obs['y']], self.COLOR_PARTICLE_BAD, 20)
                # sfx: collision_sound
                continue

            if not obs['cleared'] and obs['x'] + self.OBSTACLE_WIDTH < self.PLAYER_X:
                obs['cleared'] = True
                step_reward += 1  # Reward for clearing

                time_off_beat = min(self.steps % self.BEAT_PERIOD, self.BEAT_PERIOD - (self.steps % self.BEAT_PERIOD))
                
                if time_off_beat <= self.PERFECT_WINDOW:
                    step_reward += 5  # Bonus for perfect timing
                    self.score += 10
                    self._create_particles([self.player_pos[0] + self.PLAYER_WIDTH / 2, self.player_pos[1] + self.PLAYER_HEIGHT / 2], self.COLOR_PARTICLE_GOOD, 30)
                    # sfx: perfect_jump_sound
                else:
                    self.score += 1
                    # sfx: clear_obstacle_sound
            
            if obs['x'] < -self.OBSTACLE_WIDTH:
                self.obstacles.remove(obs)
        
        return step_reward

    def _update_particles(self):
        for p in reversed(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _spawn_obstacles(self):
        if self.steps >= self.next_obstacle_spawn_step and self.steps < self.FINISH_DISTANCE - 120:
            self.obstacles.append({
                'x': self.SCREEN_WIDTH,
                'y': self.GROUND_Y - self.OBSTACLE_HEIGHT,
                'cleared': False
            })
            self.next_obstacle_spawn_step = self.steps + self.obstacle_interval

    def _update_difficulty(self):
        if self.steps > 0:
            if self.steps % 500 == 0:
                self.obstacle_speed = min(12, self.obstacle_speed + 0.05)
            if self.steps % 250 == 0:
                self.obstacle_interval = max(45, self.obstacle_interval * 0.99)

    def _check_termination(self):
        return self.missed_beats >= 3 or self.steps >= self.MAX_STEPS or self.steps >= self.FINISH_DISTANCE

    def _get_observation(self):
        self._render_background()
        self._render_ground()
        self._render_finish_line()
        self._render_obstacles()
        self._render_player()
        self._render_particles()
        self._render_beat_indicator()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "missed_beats": self.missed_beats,
            "distance_to_finish": max(0, self.FINISH_DISTANCE - self.steps)
        }

    def _render_background(self):
        # Simple gradient
        rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        color1 = self.COLOR_BG_TOP
        color2 = self.COLOR_BG_BOTTOM
        pygame.draw.rect(self.screen, color1, rect)
        
        num_steps = 50
        for i in range(num_steps):
            inter_color = [
                color1[j] + (float(i) / num_steps) * (color2[j] - color1[j])
                for j in range(3)
            ]
            pygame.draw.line(self.screen, inter_color, (0, self.SCREEN_HEIGHT * i / num_steps), (self.SCREEN_WIDTH, self.SCREEN_HEIGHT * i / num_steps))

    def _render_ground(self):
        pygame.draw.line(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y), (self.SCREEN_WIDTH, self.GROUND_Y), 5)

    def _render_finish_line(self):
        finish_x = self.FINISH_DISTANCE - self.steps + self.PLAYER_X
        if finish_x < self.SCREEN_WIDTH + 50:
            tile_size = 20
            for i in range(int(self.SCREEN_HEIGHT / tile_size)):
                for j in range(3):
                    color = (255, 255, 255) if (i + j) % 2 == 0 else (100, 100, 100)
                    pygame.draw.rect(self.screen, color, (int(finish_x + j * tile_size), i * tile_size, tile_size, tile_size))
            pygame.draw.line(self.screen, (255,215,0), (int(finish_x), 0), (int(finish_x), self.SCREEN_HEIGHT), 5)


    def _render_player(self):
        pos_x, pos_y = int(self.player_pos[0]), int(self.player_pos[1])
        # Glow effect
        glow_radius = int(self.PLAYER_WIDTH * 1.2)
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(temp_surf, (pos_x + self.PLAYER_WIDTH // 2 - glow_radius, pos_y + self.PLAYER_HEIGHT // 2 - glow_radius))

        # Player rect
        player_rect = pygame.Rect(pos_x, pos_y, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
    
    def _render_obstacles(self):
        for obs in self.obstacles:
            pos_x, pos_y = int(obs['x']), int(obs['y'])
            # Glow effect
            glow_radius = int(self.OBSTACLE_WIDTH * 1.1)
            temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, self.COLOR_OBSTACLE_GLOW, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(temp_surf, (pos_x + self.OBSTACLE_WIDTH // 2 - glow_radius, pos_y + self.OBSTACLE_HEIGHT // 2 - glow_radius))

            # Obstacle rect
            obs_rect = pygame.Rect(pos_x, pos_y, self.OBSTACLE_WIDTH, self.OBSTACLE_HEIGHT)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs_rect, border_radius=3)

    def _render_particles(self):
        for p in self.particles:
            size = max(1, int(p['life'] * 0.4))
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), size)

    def _render_beat_indicator(self):
        progress = (self.steps % self.BEAT_PERIOD) / self.BEAT_PERIOD
        alpha = int(255 * (1 - progress) ** 2)
        size = int(10 + 30 * progress)
        
        indicator_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
        color = self.COLOR_BEAT_INDICATOR + (alpha,)
        pygame.gfxdraw.aacircle(indicator_surf, size, size, size-1, color)
        pygame.gfxdraw.filled_circle(indicator_surf, size, size, size-1, color)
        self.screen.blit(indicator_surf, (20 - size, self.GROUND_Y + 20 - size))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        # Missed Beats
        miss_text = self.font_small.render(f"MISSES: {self.missed_beats} / 3", True, self.COLOR_TEXT)
        text_rect = miss_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(miss_text, text_rect)

        # Progress Bar
        progress_ratio = self.steps / self.FINISH_DISTANCE
        bar_width = self.SCREEN_WIDTH - 40
        bar_height = 10
        pygame.draw.rect(self.screen, (50, 50, 80), (20, self.SCREEN_HEIGHT - 30, bar_width, bar_height), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (20, self.SCREEN_HEIGHT - 30, bar_width * progress_ratio, bar_height), border_radius=5)

    def _create_particles(self, pos, color, count, is_landing=False):
        for _ in range(count):
            if is_landing:
                angle = self.np_random.uniform(math.pi, 2 * math.pi)
                speed = self.np_random.uniform(1, 3)
            else:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 5)

            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'color': color})

    def validate_implementation(self):
        print("Validating implementation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to test the environment
    env = GameEnv(render_mode="rgb_array")
    
    # --- Human Play ---
    # To play the game, you need to install pygame and run this file.
    # The environment will be rendered in a pygame window.
    
    # Use a different screen for display
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Rhythm Runner")
    
    obs, info = env.reset()
    terminated = False
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            action[1] = 1 # Jump
        if keys[pygame.K_r]: # Press 'R' to reset
            obs, info = env.reset()
            terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(30) # Run at 30 FPS

    pygame.quit()