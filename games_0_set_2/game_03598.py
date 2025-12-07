
# Generated: 2025-08-27T23:50:05.131982
# Source Brief: brief_03598.md
# Brief Index: 3598

        
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
        "Controls: Press space to jump over obstacles. Timing is everything!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, side-scrolling rhythm game. Jump over neon obstacles to the beat, build combos, and race to the finish line before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.FINISH_LINE_POS = 8000
        self.MAX_STEPS = 1800 # 60 seconds at 30fps

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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_combo = pygame.font.SysFont("Consolas", 16, bold=True)

        # Colors
        self.COLOR_BG = (16, 16, 32)
        self.COLOR_GROUND = (200, 200, 220)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_PLAYER_GLOW = (0, 128, 128)
        self.OBSTACLE_COLORS = [(255, 0, 128), (255, 255, 0)]
        self.COLOR_FINISH = (255, 255, 255)
        self.COLOR_TEXT = (255, 255, 255)

        # Game constants
        self.GROUND_Y = self.HEIGHT - 50
        self.BASE_SPEED = 4.0
        self.GRAVITY = 0.8
        self.JUMP_FORCE = -15
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel_y = None
        self.on_ground = None
        self.world_scroll = None
        self.current_speed = None
        self.obstacles = None
        self.last_obstacle_x = None
        self.particles = None
        self.stars = None
        self.steps = None
        self.score = None
        self.combo = None
        self.jump_cleared_obstacle = None
        self.game_over = None
        self.np_random = None

        self.reset()
        # self.validate_implementation() # Optional: call to check your work

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = [100, self.GROUND_Y]
        self.player_vel_y = 0
        self.on_ground = True
        self.world_scroll = 0
        self.current_speed = self.BASE_SPEED
        self.obstacles = []
        self.last_obstacle_x = 400
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.combo = 0
        self.jump_cleared_obstacle = False
        self.game_over = False

        self._generate_stars()
        self._generate_obstacles(10)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        space_held = action[1] == 1
        
        reward = 0.1  # Survival reward
        
        # --- Game Logic Update ---
        
        # 1. Handle Input
        if space_held and self.on_ground:
            self.player_vel_y = self.JUMP_FORCE
            self.on_ground = False
            self.jump_cleared_obstacle = False
            self._create_particles(self.player_pos[0], self.player_pos[1] + 10, 20, self.COLOR_PLAYER, 'jump')
            # sfx: player_jump.wav
            
            # Penalty for jumping with no obstacle nearby
            imminent_obstacle = False
            for obs in self.obstacles:
                obs_screen_x = obs['x'] - self.world_scroll
                if self.player_pos[0] < obs_screen_x < self.player_pos[0] + 300:
                    imminent_obstacle = True
                    break
            if not imminent_obstacle:
                reward -= 0.2

        # 2. Update Player Physics
        if not self.on_ground:
            self.player_vel_y += self.GRAVITY
            self.player_pos[1] += self.player_vel_y
        
        if self.player_pos[1] >= self.GROUND_Y:
            self.player_pos[1] = self.GROUND_Y
            if not self.on_ground: # Just landed
                self.on_ground = True
                self.player_vel_y = 0
                self._create_particles(self.player_pos[0], self.player_pos[1] + 10, 10, self.COLOR_GROUND, 'land')
                # sfx: player_land.wav
                if not self.jump_cleared_obstacle and self.combo > 0:
                    self.combo = 0 # Reset combo on wasted jump
        
        # 3. Update World
        self.current_speed = self.BASE_SPEED + (self.steps // 200) * 0.2
        self.world_scroll += self.current_speed

        # 4. Update Obstacles
        player_rect = pygame.Rect(self.player_pos[0] - 10, self.player_pos[1] - 20, 20, 20)
        
        cleared_reward = 0
        obstacles_to_keep = []
        for obs in self.obstacles:
            obs_screen_x = obs['x'] - self.world_scroll
            if obs_screen_x < -obs['w']: # Off-screen left
                continue
            
            obs_rect = pygame.Rect(obs_screen_x, obs['y'], obs['w'], obs['h'])

            # Check for clearing an obstacle
            if not obs['cleared'] and obs_rect.right < player_rect.left:
                obs['cleared'] = True
                self.combo += 1
                cleared_reward += 1.0 + (0.5 * self.combo)
                self.jump_cleared_obstacle = True
                self._create_particles(obs_rect.centerx, obs_rect.centery, 30, obs['color'], 'clear')
                # sfx: obstacle_clear.wav
            
            # Check for collision
            if player_rect.colliderect(obs_rect):
                self.game_over = True
                reward = -10 # Terminal penalty
                self._create_particles(player_rect.centerx, player_rect.centery, 50, (255, 50, 50), 'impact')
                # sfx: player_impact.wav
                break # Stop checking after a crash

            obstacles_to_keep.append(obs)
        
        self.obstacles = obstacles_to_keep
        reward += cleared_reward

        # Procedurally generate new obstacles
        if self.last_obstacle_x - self.world_scroll < self.WIDTH + 200:
            self._generate_obstacles(1)

        # 5. Update Particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] - 0.2)

        # 6. Check Termination Conditions
        terminated = self.game_over
        
        # Reached finish line
        if self.world_scroll + self.player_pos[0] >= self.FINISH_LINE_POS:
            terminated = True
            time_factor = max(0, (self.MAX_STEPS - self.steps) / self.MAX_STEPS)
            finish_reward = 50.0 + (50.0 * time_factor)
            reward += finish_reward
            # sfx: level_complete.wav

        # Timer ran out
        if self.steps >= self.MAX_STEPS:
            terminated = True
            
        self.game_over = terminated
        self.score += reward
        self.steps += 1
        
        if self.auto_advance:
            self.clock.tick(30)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
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
            "combo": self.combo,
            "world_scroll": self.world_scroll,
        }

    def _render_game(self):
        self._render_background()
        self._render_beat_indicator()
        self._render_ground()
        self._render_finish_line()
        self._render_obstacles()
        self._render_particles()
        self._render_player()

    def _render_background(self):
        # Parallax stars
        for star in self.stars:
            screen_x = (star['x'] - self.world_scroll * star['layer']) % self.WIDTH
            pygame.gfxdraw.aacircle(self.screen, int(screen_x), int(star['y']), int(star['r']), star['c'])

    def _render_beat_indicator(self):
        # Pulse to a 120 BPM rhythm (beat every 15 frames)
        beat_progress = (self.steps % 15) / 15.0
        alpha = int(100 * (1 - beat_progress)**2)
        if alpha > 5:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            pygame.draw.rect(s, (255, 255, 255, alpha), (0, 0, self.WIDTH, self.HEIGHT), 5)
            self.screen.blit(s, (0, 0))

    def _render_ground(self):
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))
        # Add some lines for a sense of speed
        for i in range(20):
            line_x = (self.WIDTH - (self.world_scroll * 2) % 100) + i * 100
            pygame.draw.line(self.screen, self.COLOR_BG, (line_x, self.GROUND_Y), (line_x - 20, self.GROUND_Y + 15), 2)

    def _render_finish_line(self):
        finish_screen_x = self.FINISH_LINE_POS - self.world_scroll
        if 0 < finish_screen_x < self.WIDTH:
            for i in range(0, int(self.HEIGHT - self.GROUND_Y), 20):
                color1 = self.COLOR_FINISH if (i // 20) % 2 == 0 else self.COLOR_BG
                color2 = self.COLOR_BG if (i // 20) % 2 == 0 else self.COLOR_FINISH
                pygame.draw.rect(self.screen, color1, (finish_screen_x, self.GROUND_Y + i, 10, 10))
                pygame.draw.rect(self.screen, color2, (finish_screen_x + 10, self.GROUND_Y + i, 10, 10))
            pygame.draw.line(self.screen, self.COLOR_FINISH, (finish_screen_x, 0), (finish_screen_x, self.HEIGHT), 2)

    def _render_obstacles(self):
        for obs in self.obstacles:
            screen_x = obs['x'] - self.world_scroll
            rect = pygame.Rect(int(screen_x), int(obs['y']), int(obs['w']), int(obs['h']))
            pygame.draw.rect(self.screen, obs['color'], rect)
            pygame.draw.rect(self.screen, self.COLOR_BG, rect.inflate(-6, -6))
            pygame.draw.rect(self.screen, obs['color'], rect.inflate(-10, -10))

    def _render_player(self):
        x, y = int(self.player_pos[0]), int(self.player_pos[1])
        # Glow effect
        glow_size = 30 + 5 * math.sin(self.steps * 0.2)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_GLOW, (x - glow_size/2, y - 20 - (glow_size-20)/2, glow_size, glow_size), border_radius=8)
        # Player body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (x - 10, y - 20, 20, 20), border_radius=4)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color'] + (alpha,)
            pos = (int(p['pos'][0] - self.world_scroll), int(p['pos'][1]))
            
            s = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(s, int(p['radius']), int(p['radius']), int(p['radius']), color)
            self.screen.blit(s, (pos[0] - p['radius'], pos[1] - p['radius']))

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {int(self.score):05d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        if self.combo > 1:
            combo_text = self.font_combo.render(f"{self.combo}x COMBO!", True, self.COLOR_PLAYER)
            self.screen.blit(combo_text, (10, 35))

        time_left = max(0, self.MAX_STEPS - self.steps)
        time_text = self.font_ui.render(f"TIME: {time_left/30:.1f}s", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'x': self.np_random.uniform(0, self.WIDTH),
                'y': self.np_random.uniform(0, self.HEIGHT),
                'r': self.np_random.uniform(0.5, 1.5),
                'layer': self.np_random.choice([0.1, 0.3, 0.6]),
                'c': (
                    self.np_random.integers(100, 200),
                    self.np_random.integers(100, 200),
                    self.np_random.integers(150, 255)
                )
            })

    def _generate_obstacles(self, count):
        for i in range(count):
            gap = self.np_random.uniform(250, 400)
            x = self.last_obstacle_x + gap
            
            height = self.np_random.uniform(30, 80)
            width = self.np_random.uniform(20, 40)
            y = self.GROUND_Y - height
            
            self.obstacles.append({
                'x': x, 'y': y, 'w': width, 'h': height,
                'color': self.np_random.choice(self.OBSTACLE_COLORS),
                'cleared': False
            })
            self.last_obstacle_x = x

    def _create_particles(self, x, y, count, color, p_type):
        for _ in range(count):
            if p_type == 'jump' or p_type == 'land':
                vel = [self.np_random.uniform(-1, 1), self.np_random.uniform(0, 2)]
            elif p_type == 'impact':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 8)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            elif p_type == 'clear':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            else:
                vel = [0,0]

            life = self.np_random.integers(20, 40)
            self.particles.append({
                'pos': [x + self.world_scroll, y],
                'vel': vel,
                'radius': self.np_random.uniform(2, 6),
                'life': life,
                'max_life': life,
                'color': color
            })
    
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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # To control the game:
    # - `keys[pygame.K_SPACE]` for jump
    
    # Main game loop
    running = True
    while running:
        action = env.action_space.sample() # Start with random action
        action[1] = 0 # Default to not jumping
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            action[1] = 1

        if keys[pygame.K_r]: # Press R to reset
             obs, info = env.reset()

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}")
            obs, info = env.reset()

        # Pygame screen needs to be created for display
        display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        
        # Transpose back for pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
    env.close()