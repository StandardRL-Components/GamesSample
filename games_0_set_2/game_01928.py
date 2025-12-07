
# Generated: 2025-08-27T18:42:17.256737
# Source Brief: brief_01928.md
# Brief Index: 1928

        
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
        "Controls: ↑ to accelerate, ↓ to decelerate, ←→ to steer. "
        "Press space for a speed boost."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A high-speed arcade racer. Navigate a neon-lit track, "
        "avoid obstacles, and reach the finish line before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1000
        self.GAME_DURATION_SECONDS = 30
        self.FINISH_LINE_Y = -8000  # World coordinate of the finish line

        # Visuals
        self.COLOR_BG = (10, 5, 25)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_PLAYER_GLOW = (0, 255, 255, 50)
        self.COLOR_TRACK = (100, 80, 200)
        self.COLOR_TRACK_GLOW = (100, 80, 200, 30)
        self.COLOR_OBSTACLE = (255, 0, 100)
        self.COLOR_OBSTACLE_GLOW = (255, 0, 100, 50)
        self.COLOR_FINISH = (255, 223, 0)
        self.COLOR_FINISH_GLOW = (255, 223, 0, 100)
        self.COLOR_TEXT = (220, 220, 220)
        
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
        self.font_large = pygame.font.Font(None, 72)
        
        # Initialize state variables to be set in reset()
        self.player_x = 0
        self.player_vy = 0
        self.player_boost = 0
        self.world_y = 0
        self.obstacles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.time_left = 0
        self.base_obstacle_speed = 0
        self.game_over = False
        self.win_condition = False
        self.near_miss_cooldown = 0
        
        # Initialize state
        self.reset()
        
        # Validate implementation after setup
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Player state
        self.player_x = self.WIDTH // 2
        self.player_vy = 0.0  # Vertical speed
        self.player_boost = 0.0

        # World state
        self.world_y = 0.0  # Camera scroll position
        self.obstacles = []
        self.particles = []
        
        # Game state
        self.steps = 0
        self.score = 0
        self.time_left = self.GAME_DURATION_SECONDS
        self.base_obstacle_speed = 2.0
        self.game_over = False
        self.win_condition = False
        self.near_miss_cooldown = 0

        # Procedurally generate initial obstacles
        for i in range(20):
            self._spawn_obstacle(random.uniform(-self.HEIGHT * 2, 0))

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False

        if not self.game_over:
            # Unpack factorized action
            movement = action[0]
            space_pressed = action[1] == 1
            # shift_held is not used in this game

            # Update player based on action
            self._handle_input(movement, space_pressed)

            # Update game physics and world state
            self._update_physics()
            self._update_world()
            
            # Calculate rewards and check for termination
            reward = self._calculate_reward()
            terminated = self._check_termination()
            self.score += reward

        self.steps += 1
        if self.auto_advance:
            self.clock.tick(self.FPS)
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed):
        # Horizontal movement
        if movement == 3:  # Left
            self.player_x -= 6
        elif movement == 4:  # Right
            self.player_x += 6
        
        # Vertical movement (acceleration/deceleration)
        if movement == 1:  # Up
            self.player_vy += 0.5
        elif movement == 2:  # Down
            self.player_vy -= 1.0 # Braking is more effective
        
        # Boost
        if space_pressed and self.player_boost <= 0.1:
            # sfx: boost sound
            self.player_boost = 5.0
            self._spawn_particles(self.player_x, self.HEIGHT - 50, 20, self.COLOR_PLAYER, 'boost')


    def _update_physics(self):
        # Apply drag
        self.player_vy *= 0.98
        self.player_vy = max(0, self.player_vy) # Cannot go backwards
        self.player_vy = min(25, self.player_vy) # Max speed

        # Apply boost
        self.player_vy += self.player_boost
        self.player_boost *= 0.8 # Boost decays quickly

        # Update world scroll based on player speed
        self.world_y -= self.player_vy

        # Keep player within track boundaries
        track_width = self.WIDTH * 0.4
        self.player_x = np.clip(self.player_x, (self.WIDTH / 2) - track_width, (self.WIDTH / 2) + track_width)

        # Spawn player trail particles
        if self.player_vy > 1:
            self._spawn_particles(self.player_x, self.HEIGHT - 50, 1, (100, 100, 255), 'trail')


    def _update_world(self):
        # Update timer
        self.time_left -= 1.0 / self.FPS
        
        # Update obstacles
        for obs in self.obstacles:
            obs['y'] += obs['vy']
        
        # Remove off-screen obstacles and spawn new ones
        self.obstacles = [obs for obs in self.obstacles if obs['y'] - self.world_y < self.HEIGHT + 50]
        while len(self.obstacles) < 20:
            self._spawn_obstacle()
        
        # Update difficulty
        if self.steps > 0 and self.steps % 50 == 0:
            self.base_obstacle_speed += 0.05

        # Update particles
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
        if self.near_miss_cooldown > 0:
            self.near_miss_cooldown -= 1

    def _calculate_reward(self):
        reward = 0.1  # Base reward for surviving a step

        player_rect = pygame.Rect(self.player_x - 5, self.HEIGHT - 60, 10, 20)
        
        min_dist = float('inf')
        for obs in self.obstacles:
            obs_rect = pygame.Rect(obs['x'], obs['y'] - self.world_y, obs['w'], obs['h'])
            if player_rect.colliderect(obs_rect):
                # sfx: explosion
                self.game_over = True
                return -100.0 # Collision penalty
            
            # Check for near miss
            dist = math.hypot(player_rect.centerx - obs_rect.centerx, player_rect.centery - obs_rect.centery)
            min_dist = min(min_dist, dist)

        if min_dist < 50 and self.near_miss_cooldown == 0:
            # sfx: whoosh
            reward -= 5.0 # Near miss penalty
            self.near_miss_cooldown = 15 # 0.5 second cooldown
            self._spawn_particles(self.player_x, self.HEIGHT - 50, 10, self.COLOR_OBSTACLE, 'miss')

        if self.world_y < self.FINISH_LINE_Y:
            self.game_over = True
            self.win_condition = True
            if self.time_left > 0:
                # sfx: win fanfare
                return 100.0 + self.time_left # Reward for winning + time bonus
            else:
                # sfx: neutral finish sound
                return -10.0 # Penalty for finishing after time limit
        
        return reward

    def _check_termination(self):
        if self.game_over:
            return True
        if self.time_left <= 0:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render track boundaries
        track_center = self.WIDTH / 2
        track_width = self.WIDTH * 0.45
        for i in range(20):
            y = (self.steps * 5 + i * 20) % self.HEIGHT
            alpha = 255 * (1 - y / self.HEIGHT)
            
            # Left line
            pygame.draw.line(self.screen, self.COLOR_TRACK + (int(alpha),), (track_center - track_width, y), (track_center - track_width, y+2), 2)
            # Right line
            pygame.draw.line(self.screen, self.COLOR_TRACK + (int(alpha),), (track_center + track_width, y), (track_center + track_width, y+2), 2)
        
        # Render finish line
        finish_screen_y = self.FINISH_LINE_Y - self.world_y
        if 0 < finish_screen_y < self.HEIGHT:
            pulse = (math.sin(self.steps * 0.5) + 1) / 2
            alpha = 150 + int(pulse * 105)
            pygame.draw.line(self.screen, self.COLOR_FINISH + (alpha,), (track_center - track_width, finish_screen_y), (track_center + track_width, finish_screen_y), 5)

        # Render particles
        for p in self.particles:
            alpha = max(0, int(255 * (p['life'] / p['max_life'])))
            color = p['color'] + (alpha,)
            size = p['size'] * (p['life'] / p['max_life'])
            pygame.draw.circle(self.screen, color, (int(p['x']), int(p['y'])), int(size))

        # Render obstacles
        for obs in self.obstacles:
            screen_y = obs['y'] - self.world_y
            if 0 < screen_y < self.HEIGHT:
                rect = pygame.Rect(obs['x'], screen_y, obs['w'], obs['h'])
                # Glow
                glow_rect = rect.inflate(10, 10)
                s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
                pygame.draw.rect(s, self.COLOR_OBSTACLE_GLOW, s.get_rect(), border_radius=3)
                self.screen.blit(s, glow_rect.topleft)
                # Main shape
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect, border_radius=3)

        # Render player
        player_screen_y = self.HEIGHT - 60
        player_points = [
            (self.player_x, player_screen_y),
            (self.player_x - 8, player_screen_y + 20),
            (self.player_x + 8, player_screen_y + 20)
        ]
        # Glow
        s = pygame.Surface((30, 40), pygame.SRCALPHA)
        pygame.gfxdraw.aapolygon(s, [(15, 0), (7, 20), (23, 20)], self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_polygon(s, [(15, 0), (7, 20), (23, 20)], self.COLOR_PLAYER_GLOW)
        self.screen.blit(s, (self.player_x - 15, player_screen_y - 5))
        # Main shape
        pygame.gfxdraw.aapolygon(self.screen, player_points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, player_points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Timer
        time_text = f"TIME: {max(0, self.time_left):.2f}"
        time_surf = self.font_small.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (10, 10))

        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 30))

        # Speed
        speed_text = f"SPEED: {int(self.player_vy * 10)}"
        speed_surf = self.font_small.render(speed_text, True, self.COLOR_TEXT)
        self.screen.blit(speed_surf, (self.WIDTH - speed_surf.get_width() - 10, 10))

        # Game Over / Win message
        if self.game_over:
            if self.win_condition:
                msg = "FINISH!"
                color = self.COLOR_FINISH
            else:
                msg = "GAME OVER"
                color = self.COLOR_OBSTACLE
            
            msg_surf = self.font_large.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "player_speed": self.player_vy,
            "world_y": self.world_y,
        }

    def _spawn_obstacle(self, y_pos=None):
        track_width = self.WIDTH * 0.4
        track_center = self.WIDTH / 2
        
        width = self.np_random.integers(20, 60)
        x = self.np_random.uniform(track_center - track_width, track_center + track_width - width)
        y = y_pos if y_pos is not None else self.world_y - self.np_random.uniform(self.HEIGHT, self.HEIGHT * 2)
        
        self.obstacles.append({
            'x': x,
            'y': y,
            'w': width,
            'h': 20,
            'vy': self.np_random.uniform(0, self.base_obstacle_speed)
        })

    def _spawn_particles(self, x, y, count, color, p_type):
        for _ in range(count):
            if p_type == 'trail':
                vx = self.np_random.uniform(-0.5, 0.5)
                vy = self.np_random.uniform(1, 3)
                life = self.np_random.integers(10, 20)
                size = self.np_random.uniform(1, 3)
            elif p_type == 'boost':
                angle = self.np_random.uniform(math.pi * 0.4, math.pi * 0.6)
                speed = self.np_random.uniform(4, 8)
                vx = math.cos(angle) * speed
                vy = math.sin(angle) * speed
                life = self.np_random.integers(20, 40)
                size = self.np_random.uniform(2, 5)
            elif p_type == 'miss':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 5)
                vx = math.cos(angle) * speed
                vy = math.sin(angle) * speed
                life = self.np_random.integers(15, 30)
                size = self.np_random.uniform(1, 4)

            self.particles.append({
                'x': x, 'y': y, 'vx': vx, 'vy': vy, 'life': life, 'max_life': life,
                'color': color, 'size': size
            })
            
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
    
    # Setup Pygame window for human play
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    done = False
    total_reward = 0
    
    # Game loop
    while not done:
        # Action mapping for human control
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Environment step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

    print(f"Game Over! Final Score: {total_reward:.2f}")
    env.close()