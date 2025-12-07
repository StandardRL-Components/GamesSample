
# Generated: 2025-08-28T04:39:27.577344
# Source Brief: brief_05318.md
# Brief Index: 5318

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Press SPACE to jump over obstacles. Time your jumps to build your combo!"

    # Must be a short, user-facing description of the game:
    game_description = "A fast-paced rhythm runner. Jump over obstacles to the beat in a vibrant, neon world."

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30 # Corresponds to step rate

        # Colors
        self.COLOR_BG_TOP = pygame.Color("#1a001a")
        self.COLOR_BG_BOTTOM = pygame.Color("#0d000d")
        self.COLOR_ROAD = pygame.Color("#2c003e")
        self.COLOR_ROAD_LINE = pygame.Color("#ff00ff")
        self.COLOR_PLAYER = pygame.Color("#00aaff")
        self.COLOR_PLAYER_GLOW = pygame.Color("#00aaff")
        self.COLOR_OBSTACLE = pygame.Color("#ff3333")
        self.COLOR_OBSTACLE_GLOW = pygame.Color("#ff3333")
        self.COLOR_TEXT = pygame.Color("#ffffff")
        self.COLOR_GOLD = pygame.Color("#ffd700")
        self.COLOR_BEAT_INDICATOR = pygame.Color("#ffffff")

        # Game parameters
        self.GROUND_Y = self.HEIGHT - 80
        self.PLAYER_X_POS = 100
        self.PLAYER_RADIUS = 15
        self.GRAVITY = 0.8
        self.JUMP_STRENGTH = -15
        self.WORLD_SPEED = 8.0
        self.STEPS_PER_BEAT = 10
        self.MAX_BEATS = 100
        self.MAX_STEPS = self.MAX_BEATS * self.STEPS_PER_BEAT

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_big = pygame.font.SysFont("Consolas", 64, bold=True)
        
        # --- State Variables (initialized in reset) ---
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_y = 0
        self.player_vy = 0
        self.is_jumping = False
        self.lives = 0
        self.combo = 0
        self.beats_survived = 0
        self.obstacles = []
        self.particles = []
        self.road_lines = []
        self.stars = []
        self.cityscape = []
        self.obstacle_spawn_chance = 0.0
        self.win = False

        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        # Player state
        self.player_y = self.GROUND_Y
        self.player_vy = 0
        self.is_jumping = False
        
        # Game progress state
        self.lives = 3
        self.combo = 0
        self.beats_survived = 0
        
        # Entity lists
        self.obstacles = []
        self.particles = []
        
        # Difficulty
        self.obstacle_spawn_chance = 0.4
        
        # Procedurally generate background
        self._generate_background()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def _generate_background(self):
        # Generate stars for parallax effect
        self.stars = [
            {
                'pos': [self.np_random.random() * self.WIDTH, self.np_random.random() * self.HEIGHT],
                'speed': 1 + self.np_random.random() * 2,
                'radius': self.np_random.random() * 1.5
            } for _ in range(100)
        ]
        # Generate distant cityscape silhouette
        self.cityscape = []
        x = 0
        while x < self.WIDTH:
            w = self.np_random.integers(30, 80)
            h = self.np_random.integers(50, 200)
            self.cityscape.append(pygame.Rect(x, self.HEIGHT - h, w, h))
            x += w + self.np_random.integers(5, 20)
            
        # Generate road lines for speed effect
        self.road_lines = [i * 80 for i in range(self.WIDTH // 80 + 2)]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # Not used
        space_pressed = action[1] == 1
        shift_held = action[2] == 1 # Not used
        
        # Update game logic
        reward = 0
        
        # Player physics
        if space_pressed and not self.is_jumping:
            self.player_vy = self.JUMP_STRENGTH
            self.is_jumping = True
            # SFX: Jump sound
            self._create_particles((self.PLAYER_X_POS, self.player_y), self.COLOR_PLAYER, 10, speed_mult=0.5)

        self.player_vy += self.GRAVITY
        self.player_y += self.player_vy
        
        if self.player_y >= self.GROUND_Y:
            self.player_y = self.GROUND_Y
            self.player_vy = 0
            if self.is_jumping: # Landing
                self.is_jumping = False
                # SFX: Land sound
                self._create_particles((self.PLAYER_X_POS, self.player_y + self.PLAYER_RADIUS), self.COLOR_ROAD_LINE, 5, speed_mult=0.3, life=10)

        # Update all dynamic elements
        self._update_background()
        self._update_obstacles()
        self._update_particles()
        
        # Handle interactions
        reward += self._handle_collisions_and_scoring()
        
        # Update game state
        self.steps += 1
        is_beat_frame = self.steps % self.STEPS_PER_BEAT == 0
        if is_beat_frame:
            self.beats_survived += 1
            # Increase difficulty every 10 beats
            if self.beats_survived > 0 and self.beats_survived % 10 == 0:
                self.obstacle_spawn_chance = min(0.9, self.obstacle_spawn_chance + 0.02)
        
        terminated = self._check_termination()

        if self.win and not self.game_over:
             reward += 50 # Win bonus
             # SFX: Level complete fanfare
             self._create_particles((self.WIDTH/2, self.HEIGHT/2), self.COLOR_GOLD, 100, life=120, speed_mult=3)
        
        if terminated:
            self.game_over = True

        self.score += reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _update_background(self):
        for i in range(len(self.road_lines)):
            self.road_lines[i] -= self.WORLD_SPEED
            if self.road_lines[i] < -80:
                self.road_lines[i] += (len(self.road_lines)) * 80
        for star in self.stars:
            star['pos'][0] -= star['speed'] * 0.1
            if star['pos'][0] < 0:
                star['pos'][0] = self.WIDTH
                star['pos'][1] = self.np_random.random() * self.HEIGHT

    def _update_obstacles(self):
        for o in self.obstacles:
            o['x'] -= self.WORLD_SPEED
        self.obstacles = [o for o in self.obstacles if o['x'] + o['w'] > 0]
        
        is_beat_frame = self.steps % self.STEPS_PER_BEAT == 0
        if is_beat_frame:
            can_spawn = not self.obstacles or self.obstacles[-1]['x'] < self.WIDTH - 150
            if can_spawn and self.np_random.random() < self.obstacle_spawn_chance:
                h = self.np_random.integers(25, 70)
                w = self.np_random.integers(30, 80)
                self.obstacles.append({'x': self.WIDTH, 'y': self.GROUND_Y - h, 'w': w, 'h': h, 'scored': False})

    def _handle_collisions_and_scoring(self):
        reward = 0
        player_rect = pygame.Rect(self.PLAYER_X_POS - self.PLAYER_RADIUS, self.player_y - self.PLAYER_RADIUS, self.PLAYER_RADIUS * 2, self.PLAYER_RADIUS * 2)

        for o in self.obstacles:
            obstacle_rect = pygame.Rect(o['x'], o['y'], o['w'], o['h'])
            
            if player_rect.colliderect(obstacle_rect):
                if not o.get('hit', False):
                    o['hit'] = True
                    self.lives -= 1
                    reward = -10 # Explicit penalty for collision
                    self.combo = 0
                    # SFX: Collision/Explosion
                    self._create_particles(player_rect.center, self.COLOR_OBSTACLE, 30)
                continue

            if not o['scored'] and o['x'] + o['w'] < self.PLAYER_X_POS:
                o['scored'] = True
                self.combo += 1
                if self.is_jumping or self.player_y < self.GROUND_Y - 1:
                    reward += 5 # Perfect jump reward
                    # SFX: Success Chime (gold)
                    self._create_particles((self.PLAYER_X_POS, self.player_y), self.COLOR_GOLD, 15, life=20)
                else:
                    reward += 1 # Base reward for passing under
        return reward

    def _check_termination(self):
        if self.lives <= 0:
            return True
        if self.beats_survived >= self.MAX_BEATS:
            self.win = True
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        # Clear screen and render all elements
        self._render_all()
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "combo": self.combo,
            "beats_survived": self.beats_survived,
        }
        
    def _render_all(self):
        self._draw_gradient_background()
        self._draw_stars()
        self._draw_cityscape()
        self._draw_road()
        self._draw_obstacles()
        self._draw_player()
        self._draw_particles()
        self._render_ui()
        
    def _draw_gradient_background(self):
        for y in range(self.HEIGHT):
            color = self.COLOR_BG_TOP.lerp(self.COLOR_BG_BOTTOM, y / self.HEIGHT)
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _draw_stars(self):
        for star in self.stars:
            pos = (int(star['pos'][0]), int(star['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(star['radius']), (200, 200, 255, 150))
            
    def _draw_cityscape(self):
        for building in self.cityscape:
            pygame.draw.rect(self.screen, (20, 0, 30), building)

    def _draw_road(self):
        pygame.draw.rect(self.screen, self.COLOR_ROAD, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))
        pygame.draw.line(self.screen, self.COLOR_ROAD_LINE, (0, self.GROUND_Y), (self.WIDTH, self.GROUND_Y), 3)
        for x in self.road_lines:
            pygame.draw.line(self.screen, self.COLOR_ROAD_LINE, (int(x), self.GROUND_Y + 20), (int(x) - 40, self.HEIGHT), 2)
            
    def _draw_obstacles(self):
        for o in self.obstacles:
            rect = pygame.Rect(int(o['x']), int(o['y']), int(o['w']), int(o['h']))
            glow_rect = rect.inflate(8, 8)
            glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, self.COLOR_OBSTACLE_GLOW, glow_surf.get_rect(), border_radius=5)
            glow_surf.set_alpha(80)
            self.screen.blit(glow_surf, glow_rect.topleft)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect, border_radius=3)

    def _draw_player(self):
        pos = (int(self.PLAYER_X_POS), int(self.player_y))
        for i in range(4):
            alpha = 80 - i * 20
            radius = self.PLAYER_RADIUS + i * 4
            color = (*self.COLOR_PLAYER_GLOW[:3], alpha)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _render_ui(self):
        combo_text = self.font_ui.render(f"COMBO: {self.combo}x", True, self.COLOR_TEXT)
        self.screen.blit(combo_text, (10, 10))
        lives_text = self.font_ui.render(f"LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - 150, 10))
        for i in range(self.lives):
            pos = (self.WIDTH - 70 + i * 25, 22)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_PLAYER)

        is_beat_frame = self.steps % self.STEPS_PER_BEAT == 0
        beat_alpha = 255 if is_beat_frame else 50
        beat_radius = 15 if is_beat_frame else 10
        beat_color = (*self.COLOR_BEAT_INDICATOR[:3], beat_alpha)
        pygame.gfxdraw.filled_circle(self.screen, self.WIDTH // 2, self.HEIGHT - 20, beat_radius, beat_color)
        
        progress = self.beats_survived / self.MAX_BEATS
        bar_width = self.WIDTH * 0.8
        bar_x = self.WIDTH * 0.1
        pygame.draw.rect(self.screen, (255, 255, 255, 50), (bar_x, self.HEIGHT - 45, bar_width, 5), border_radius=2)
        pygame.draw.rect(self.screen, self.COLOR_GOLD, (bar_x, self.HEIGHT - 45, bar_width * progress, 5), border_radius=2)
        
        if self.game_over:
            msg = "LEVEL COMPLETE" if self.win else "GAME OVER"
            color = self.COLOR_GOLD if self.win else self.COLOR_OBSTACLE
            text_surf = self.font_big.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _create_particles(self, pos, color, count, speed_mult=1.0, life=30):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = (1 + self.np_random.random() * 3) * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({'pos': list(pos), 'vel': vel, 'radius': self.np_random.random() * 3 + 1, 'color': color, 'life': self.np_random.integers(life // 2, life), 'max_life': life})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        
    def _draw_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'][:3], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['radius'] * (p['life'] / p['max_life']))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

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

if __name__ == '__main__':
    # This block allows for both headless testing and visualization.
    # To run headlessly (e.g., in a server environment), keep the dummy driver.
    # To visualize, comment out the os.environ line and ensure you have a display.
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    
    try:
        # --- Visualization setup ---
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Rhythm Runner")
        clock = pygame.time.Clock()
        
        obs, info = env.reset()
        done = False
        
        print(env.user_guide)
        
        # --- Game Loop ---
        while not done:
            # Simple human input mapping for visualization
            keys = pygame.key.get_pressed()
            action = [0, 0, 0] # no-op
            if keys[pygame.K_SPACE]:
                action[1] = 1 # Jump

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Pygame event handling for quitting
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            # Render the observation from the env to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated:
                print(f"Episode finished. Final Info: {info}")
                pygame.time.wait(2000) # Pause before reset
                obs, info = env.reset()
        
            clock.tick(env.FPS)

    except pygame.error as e:
        print("\nPygame display error. This is expected if you are running in a headless environment.")
        print("To test the logic without visualization, the script can be run as is.")
        # Fallback to headless test
        env.reset()
        for i in range(100):
            action = env.action_space.sample()
            _, _, terminated, _, _ = env.step(action)
            if terminated:
                env.reset()
        print("\nHeadless test completed: 100 steps with random actions ran without crashing.")

    finally:
        env.close()