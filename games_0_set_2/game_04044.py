
# Generated: 2025-08-28T01:13:56.558279
# Source Brief: brief_04044.md
# Brief Index: 4044

        
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

    user_guide = (
        "Controls: ↑ to Jump, ↓ to Slide. Avoid the red obstacles."
    )

    game_description = (
        "A fast-paced side-scrolling runner. Time your jumps and slides to "
        "navigate through increasingly difficult stages and reach the finish line."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.render_mode = render_mode

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GROUND_Y = 320
        self.PLAYER_X = 120
        self.NUM_STAGES = 3
        self.STAGE_LENGTH_FRAMES = 50 * self.FPS  # 50 seconds per stage

        # --- Physics Constants ---
        self.GRAVITY = 0.8
        self.JUMP_STRENGTH = -16
        self.SLIDE_DURATION_FRAMES = 20
        self.INVINCIBILITY_FRAMES = 60

        # --- Visuals ---
        self.FONT_LARGE = pygame.font.Font(None, 48)
        self.FONT_MEDIUM = pygame.font.Font(None, 32)
        self.FONT_SMALL = pygame.font.Font(None, 24)

        self.COLOR_BG_TOP = (4, 12, 48)
        self.COLOR_BG_BOTTOM = (24, 48, 112)
        self.COLOR_GROUND = (34, 68, 132)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_GLOW = (0, 255, 150, 50)
        self.COLOR_OBSTACLE = (255, 50, 50)
        self.COLOR_OBSTACLE_GLOW = (255, 50, 50, 70)
        self.COLOR_PARTICLE_ACTION = (220, 255, 240)
        self.COLOR_PARTICLE_CRASH = (255, 150, 50)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_GOLD = (255, 215, 0)
        
        self._create_background_surface()

        # --- State Variables ---
        # These are initialized in reset()
        self.seed = None
        self.player_y = 0
        self.player_vy = 0
        self.is_jumping = False
        self.is_sliding = False
        self.slide_timer = 0
        self.player_trail = []

        self.obstacles = []
        self.obstacle_speed = 0
        self.obstacle_spawn_timer = 0

        self.particles = []
        
        self.stage = 0
        self.stage_progress = 0
        self.score = 0
        self.lives = 0
        self.invincibility_timer = 0

        self.game_over = False
        self.win = False

        self.reset()
        # self.validate_implementation() # For development

    def _create_background_surface(self):
        self.bg_surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.bg_surface, color, (0, y), (self.WIDTH, y))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            self.seed = seed

        self.score = 0
        self.lives = 3
        self.game_over = False
        self.win = False
        
        self._start_stage(initial_stage=1)
        
        return self._get_observation(), self._get_info()

    def _start_stage(self, initial_stage=None):
        if initial_stage is not None:
            self.stage = initial_stage
        
        self.player_y = self.GROUND_Y
        self.player_vy = 0
        self.is_jumping = False
        self.is_sliding = False
        self.slide_timer = 0
        self.player_trail = []
        
        self.stage_progress = 0
        self.obstacles = []
        self.particles = []
        self.invincibility_timer = self.FPS # Brief grace period at stage start

        # Difficulty scaling
        self.obstacle_speed = 6 + self.stage * 1.5
        min_gap = max(30, 70 - self.stage * 10)
        self.obstacle_spawn_timer = random.randint(min_gap, min_gap + 40)

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            # If the game is over, just return the final state without updates
            reward = 0
            terminated = True
            return self._get_observation(), reward, terminated, False, self._get_info()

        movement = action[0]
        reward = 0.0

        # --- Update Game Logic ---
        self._update_player(movement)
        cleared_reward = self._update_world()
        self._update_particles()
        
        # --- Rewards & Penalties ---
        reward += 0.01  # Small reward for surviving a frame

        # Penalize using an action when no obstacle is near
        if movement in [1, 2] and self._is_action_safe():
            reward -= 0.02
        
        reward += cleared_reward

        # --- Collision ---
        if self.invincibility_timer <= 0:
            if self._handle_collisions():
                self.lives -= 1
                self.invincibility_timer = self.INVINCIBILITY_FRAMES
                # sfx: player_hit
                if self.lives > 0:
                    self._create_particles(self.PLAYER_X, self.player_y, 20, self.COLOR_PARTICLE_CRASH)

        # --- Update Timers ---
        self.stage_progress += 1
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1

        # --- Check Termination Conditions ---
        terminated = False
        if self.lives <= 0:
            terminated = True
            self.game_over = True
            reward = -100
            self._create_particles(self.PLAYER_X, self.player_y, 50, self.COLOR_PARTICLE_CRASH)
        elif self.stage_progress >= self.STAGE_LENGTH_FRAMES:
            reward += 5  # Stage complete reward
            self.stage += 1
            if self.stage > self.NUM_STAGES:
                self.win = True
                terminated = True
                self.game_over = True
                reward += 100  # Win bonus
            else:
                self._start_stage()
                # sfx: stage_clear

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, movement):
        # Handle slide
        if movement == 2 and not self.is_jumping: # Slide
            if not self.is_sliding:
                self.is_sliding = True
                self.slide_timer = self.SLIDE_DURATION_FRAMES
                # sfx: slide_start
                self._create_particles(self.PLAYER_X, self.GROUND_Y, 5, self.COLOR_PARTICLE_ACTION, direction='horizontal')

        if self.is_sliding:
            self.slide_timer -= 1
            if self.slide_timer <= 0:
                self.is_sliding = False

        # Handle jump
        if movement == 1 and not self.is_jumping and not self.is_sliding: # Jump
            self.is_jumping = True
            self.player_vy = self.JUMP_STRENGTH
            # sfx: jump
            self._create_particles(self.PLAYER_X, self.GROUND_Y, 10, self.COLOR_PARTICLE_ACTION, direction='down')

        # Apply gravity
        self.player_vy += self.GRAVITY
        self.player_y += self.player_vy

        # Ground collision
        if self.player_y >= self.GROUND_Y:
            self.player_y = self.GROUND_Y
            self.player_vy = 0
            if self.is_jumping:
                self.is_jumping = False
                # sfx: land
                self._create_particles(self.PLAYER_X, self.GROUND_Y, 5, self.COLOR_PARTICLE_ACTION, direction='horizontal')

        # Update trail
        self.player_trail.append(self.player_y)
        if len(self.player_trail) > 10:
            self.player_trail.pop(0)

    def _update_world(self):
        # --- Update Obstacles ---
        cleared_obstacle_reward = 0
        for obstacle in self.obstacles:
            obstacle['x'] -= self.obstacle_speed
            if not obstacle['cleared'] and obstacle['x'] + obstacle['w'] < self.PLAYER_X:
                obstacle['cleared'] = True
                cleared_obstacle_reward += 1
                # sfx: obstacle_clear

        self.obstacles = [obs for obs in self.obstacles if obs['x'] > -50]

        # --- Spawn Obstacles ---
        self.obstacle_spawn_timer -= 1
        if self.obstacle_spawn_timer <= 0:
            obstacle_type = random.choice(['bottom', 'top'])
            new_obstacle = {'cleared': False}
            if obstacle_type == 'bottom':
                h = random.randint(30, 70)
                new_obstacle.update({'x': self.WIDTH, 'y': self.GROUND_Y - h, 'w': 25, 'h': h, 'type': 'rect'})
            else: # 'top'
                h = random.randint(40, 80)
                new_obstacle.update({'x': self.WIDTH, 'y': self.GROUND_Y - 120, 'w': 40, 'h': h, 'type': 'tri'})
            
            self.obstacles.append(new_obstacle)
            
            min_gap = max(25, 60 - self.stage * 8)
            max_gap = max(45, 80 - self.stage * 8)
            self.obstacle_spawn_timer = random.randint(min_gap, max_gap)
        
        return cleared_obstacle_reward

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_player_rect(self):
        if self.is_sliding:
            return pygame.Rect(self.PLAYER_X - 15, self.GROUND_Y - 20, 30, 20)
        else:
            return pygame.Rect(self.PLAYER_X - 5, self.player_y - 40, 10, 40)

    def _handle_collisions(self):
        player_rect = self._get_player_rect()
        for obstacle in self.obstacles:
            if obstacle['type'] == 'rect':
                obs_rect = pygame.Rect(obstacle['x'], obstacle['y'], obstacle['w'], obstacle['h'])
            else: # 'tri'
                # Simplified triangular hitbox
                obs_rect = pygame.Rect(obstacle['x'], obstacle['y'], obstacle['w'], obstacle['h'])
            
            if player_rect.colliderect(obs_rect):
                return True
        return False

    def _is_action_safe(self):
        for obs in self.obstacles:
            if obs['x'] > self.PLAYER_X and obs['x'] < self.PLAYER_X + 300:
                return False
        return True
    
    def _create_particles(self, x, y, count, color, direction=None):
        for _ in range(count):
            if direction == 'horizontal':
                angle = random.uniform(math.pi * 0.8, math.pi * 1.2)
            elif direction == 'down':
                angle = random.uniform(math.pi * 0.2, math.pi * 0.8)
            else:
                angle = random.uniform(0, 2 * math.pi)
            
            speed = random.uniform(1, 5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = random.randint(10, 25)
            radius = random.uniform(1, 4)
            self.particles.append({'pos': [x, y], 'vel': vel, 'life': life, 'radius': radius, 'color': color})

    def _get_observation(self):
        self.screen.blit(self.bg_surface, (0, 0))
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "stage": self.stage,
            "lives": self.lives,
            "stage_progress": self.stage_progress / self.STAGE_LENGTH_FRAMES
        }

    def _render_game(self):
        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))

        # Finish Line
        progress_to_end = self.stage_progress / self.STAGE_LENGTH_FRAMES
        if progress_to_end > 0.9:
            finish_x = self.WIDTH + (self.STAGE_LENGTH_FRAMES - self.stage_progress) * self.obstacle_speed
            if finish_x < self.WIDTH + 50:
                for y in range(0, self.HEIGHT, 20):
                    c = (y // 20) % 2
                    color = (255, 255, 255) if c == 0 else (0,0,0)
                    pygame.draw.rect(self.screen, color, (int(finish_x), y, 10, 20))
                pygame.draw.rect(self.screen, self.COLOR_GOLD, (int(finish_x) - 4, 0, 4, self.HEIGHT))

        # Obstacles
        for obs in self.obstacles:
            glow_rect = pygame.Rect(int(obs['x'])-5, int(obs['y'])-5, int(obs['w'])+10, int(obs['h'])+10)
            pygame.gfxdraw.box(self.screen, glow_rect, self.COLOR_OBSTACLE_GLOW)
            if obs['type'] == 'rect':
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, (int(obs['x']), int(obs['y']), int(obs['w']), int(obs['h'])))
            else: # tri
                points = [
                    (int(obs['x']), int(obs['y'] + obs['h'])),
                    (int(obs['x'] + obs['w'] / 2), int(obs['y'])),
                    (int(obs['x'] + obs['w']), int(obs['y'] + obs['h']))
                ]
                pygame.draw.polygon(self.screen, self.COLOR_OBSTACLE, points)

        # Player
        if self.lives > 0:
            # Trail
            for i, y_pos in enumerate(self.player_trail):
                alpha = (i / len(self.player_trail)) * 100
                color = (*self.COLOR_PLAYER, int(alpha))
                if self.is_sliding:
                    rect = pygame.Rect(int(self.PLAYER_X - 15 - i*0.5), int(self.GROUND_Y - 20), int(30 - i), 20)
                else:
                    rect = pygame.Rect(int(self.PLAYER_X - 5 - i*0.5), int(y_pos - 40), int(10-i*0.5), 40)
                
                if rect.width > 0 and rect.height > 0:
                    s = pygame.Surface(rect.size, pygame.SRCALPHA)
                    s.fill(color)
                    self.screen.blit(s, rect.topleft)

            # Main Body
            player_rect = self._get_player_rect()
            if self.invincibility_timer > 0 and (self.invincibility_timer // 3) % 2 == 0:
                pass # Flicker effect
            else:
                glow_rect = player_rect.inflate(20, 20)
                pygame.gfxdraw.box(self.screen, glow_rect, self.COLOR_PLAYER_GLOW)
                pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
        # Particles
        for p in self.particles:
            alpha = (p['life'] / 25)
            color = (*p['color'], int(alpha * 255))
            s = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(s, (int(p['pos'][0] - p['radius']), int(p['pos'][1] - p['radius'])))

    def _render_ui(self):
        # Score
        score_text = self.FONT_MEDIUM.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Stage
        stage_text = self.FONT_MEDIUM.render(f"STAGE: {self.stage}/{self.NUM_STAGES}", True, self.COLOR_TEXT)
        stage_rect = stage_text.get_rect(center=(self.WIDTH // 2, 25))
        self.screen.blit(stage_text, stage_rect)

        # Lives
        for i in range(self.lives):
            heart_pos = (self.WIDTH - 30 - i * 35, 15)
            points = [
                (heart_pos[0] + 12, heart_pos[1] + 5), (heart_pos[0] + 24, heart_pos[1] + 15),
                (heart_pos[0] + 12, heart_pos[1] + 25), (heart_pos[0], heart_pos[1] + 15)
            ]
            pygame.draw.polygon(self.screen, self.COLOR_OBSTACLE, points)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_OBSTACLE)

        # Progress Bar
        progress_ratio = self.stage_progress / self.STAGE_LENGTH_FRAMES
        bar_width = self.WIDTH - 20
        pygame.draw.rect(self.screen, (255,255,255,50), (10, self.HEIGHT - 20, bar_width, 10), 1)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (10, self.HEIGHT - 20, bar_width * progress_ratio, 10))

        # Game Over / Win Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_GOLD if self.win else self.COLOR_OBSTACLE
            end_text = self.FONT_LARGE.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, end_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        assert "score" in info
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        assert "score" in info

        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    
    # Use a Pygame window for human rendering
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Line Runner")
    clock = pygame.time.Clock()

    total_reward = 0
    
    while not done:
        # Human controls
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]:
            movement = 1 # jump
        elif keys[pygame.K_DOWN]:
            movement = 2 # slide
        
        # The other actions are not used for human play
        space_held = 0
        shift_held = 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}")
            # Wait a bit before closing
            pygame.time.wait(2000)
            done = True
            
        clock.tick(env.FPS)
        
    env.close()