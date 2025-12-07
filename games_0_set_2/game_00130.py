
# Generated: 2025-08-27T12:41:14.413109
# Source Brief: brief_00130.md
# Brief Index: 130

        
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
        "Controls: Space to jump. ↑/↓ for a higher jump, ←/→ for a lower jump."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Escape an alien planet by jumping over procedurally generated obstacles to reach your spaceship."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400

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
        
        # Fonts and Colors
        self.FONT_UI = pygame.font.SysFont("monospace", 20, bold=True)
        self.COLOR_BG_TOP = (20, 0, 40)
        self.COLOR_BG_BOTTOM = (0, 0, 0)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_OBSTACLE = (255, 50, 50)
        self.COLOR_PARTICLE = (255, 255, 0)
        self.COLOR_UI = (220, 220, 220)
        self.COLOR_SHIP_BODY = (180, 180, 200)
        self.COLOR_SHIP_WINDOW = (100, 200, 255)
        self.COLOR_SHIP_FIN = (140, 140, 160)

        # Game constants
        self.MAX_STEPS = 3000  # 30 seconds at 100 steps/sec
        self.GROUND_Y = self.HEIGHT - 50
        self.GRAVITY = 0.4
        self.PLAYER_X = 100
        self.PLAYER_SIZE = 15

        self.JUMP_POWER_DEFAULT = -10
        self.JUMP_POWER_HIGH = -12
        self.JUMP_POWER_LOW = -8

        self.INITIAL_OBSTACLE_SPEED = 4.0
        self.OBSTACLE_SPEED_INCREASE = 0.1
        
        # State variables will be initialized in reset()
        self.player_pos = None
        self.player_vel_y = None
        self.is_grounded = None
        self.obstacles = None
        self.particles = None
        self.obstacle_speed = None
        self.next_obstacle_x = None
        self.obstacle_id_counter = None
        self.cleared_obstacles = None
        
        self.steps = None
        self.score = None
        self.game_over = None

        # Pre-render the background for efficiency
        self.background_surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        self._draw_background_gradient()

        # Initialize state
        self.reset()
        
        # Run self-check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = [self.PLAYER_X, self.GROUND_Y]
        self.player_vel_y = 0
        self.is_grounded = True
        
        self.obstacles = []
        self.particles = []
        
        self.obstacle_speed = self.INITIAL_OBSTACLE_SPEED
        self.next_obstacle_x = self.WIDTH + 100
        self.obstacle_id_counter = 0
        self.cleared_obstacles = set()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Generate initial obstacles
        for _ in range(3):
            self._generate_obstacle()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0

        # --- 1. Handle Input ---
        movement, space_pressed, _ = action
        jump_modifier_used = movement != 0
        
        if self.is_grounded and space_pressed:
            # Sound: Player Jump
            if movement in [1, 2]:  # Higher jump
                self.player_vel_y = self.JUMP_POWER_HIGH
            elif movement in [3, 4]:  # Lower jump
                self.player_vel_y = self.JUMP_POWER_LOW
            else:  # Default jump
                self.player_vel_y = self.JUMP_POWER_DEFAULT
            self.is_grounded = False
            
            if jump_modifier_used:
                reward -= 0.2

        # --- 2. Update Game Logic ---
        self.steps += 1
        reward += 0.1 # Survival reward

        # Update player physics
        self.player_vel_y += self.GRAVITY
        self.player_pos[1] += self.player_vel_y
        
        if self.player_pos[1] >= self.GROUND_Y:
            self.player_pos[1] = self.GROUND_Y
            self.player_vel_y = 0
            self.is_grounded = True

        # Update obstacle speed
        if self.steps > 0 and self.steps % 100 == 0:
            self.obstacle_speed += self.OBSTACLE_SPEED_INCREASE

        # Update obstacles
        player_rect = self._get_player_rect()
        new_obstacles = []
        for obs in self.obstacles:
            obs['x'] -= self.obstacle_speed
            if obs['x'] + obs['width'] > 0:
                new_obstacles.append(obs)
                
                obs_rect = pygame.Rect(obs['x'], obs['y'], obs['width'], obs['height'])
                
                # Check for termination (collision)
                if player_rect.colliderect(obs_rect):
                    # Sound: Player Explosion/Hit
                    self.game_over = True
                    reward -= 100
                    self._create_explosion(player_rect.center)
                    break
                
                # Check for clearing an obstacle
                if obs['id'] not in self.cleared_obstacles and player_rect.left > obs_rect.right:
                    # Sound: Point Scored
                    reward += 10
                    self.cleared_obstacles.add(obs['id'])
                    
                # Check for near miss
                near_miss_rect = obs_rect.inflate(10, 10)
                if not player_rect.colliderect(obs_rect) and player_rect.colliderect(near_miss_rect):
                    reward += 0.5
                    # Sound: Spark/Near Miss
                    self._create_sparks(player_rect, obs_rect)
        
        self.obstacles = new_obstacles

        # Generate new obstacles if needed
        if not self.obstacles or self.obstacles[-1]['x'] < self.WIDTH - 200:
             self._generate_obstacle()

        # Update particles
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # Particle gravity
            p['lifespan'] -= 1

        # --- 3. Check for Win/Loss Conditions ---
        terminated = self.game_over
        if not terminated and self.steps >= self.MAX_STEPS:
            # Sound: Victory Fanfare
            terminated = True
            self.game_over = True
            reward += 100
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with pre-rendered background
        self.screen.blit(self.background_surface, (0, 0))
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": (self.MAX_STEPS - self.steps) / 100.0,
        }

    def _render_game(self):
        self._draw_spaceship()
        self._draw_obstacles()
        self._draw_particles()
        if not self.game_over:
            self._draw_player()

    def _render_ui(self):
        time_left = max(0, (self.MAX_STEPS - self.steps) / 100)
        timer_text = self.FONT_UI.render(f"TIME: {time_left:.2f}", True, self.COLOR_UI)
        self.screen.blit(timer_text, (10, 10))

        score_text = self.FONT_UI.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

    def _draw_background_gradient(self):
        for y in range(self.HEIGHT):
            # Interpolate color from top to bottom
            r = self.COLOR_BG_TOP[0] + (self.COLOR_BG_BOTTOM[0] - self.COLOR_BG_TOP[0]) * y / self.HEIGHT
            g = self.COLOR_BG_TOP[1] + (self.COLOR_BG_BOTTOM[1] - self.COLOR_BG_TOP[1]) * y / self.HEIGHT
            b = self.COLOR_BG_TOP[2] + (self.COLOR_BG_BOTTOM[2] - self.COLOR_BG_TOP[2]) * y / self.HEIGHT
            pygame.draw.line(self.background_surface, (r, g, b), (0, y), (self.WIDTH, y))

    def _draw_spaceship(self):
        ship_x = self.WIDTH - 40
        ship_y = self.GROUND_Y - 50
        
        # Main body
        body_points = [(ship_x, ship_y), (ship_x + 30, ship_y + 25), (ship_x, ship_y + 50)]
        pygame.gfxdraw.aapolygon(self.screen, body_points, self.COLOR_SHIP_BODY)
        pygame.gfxdraw.filled_polygon(self.screen, body_points, self.COLOR_SHIP_BODY)
        
        # Fins
        fin1_points = [(ship_x - 10, ship_y - 10), (ship_x, ship_y), (ship_x, ship_y + 10)]
        fin2_points = [(ship_x - 10, ship_y + 60), (ship_x, ship_y + 50), (ship_x, ship_y + 40)]
        pygame.gfxdraw.aapolygon(self.screen, fin1_points, self.COLOR_SHIP_FIN)
        pygame.gfxdraw.filled_polygon(self.screen, fin1_points, self.COLOR_SHIP_FIN)
        pygame.gfxdraw.aapolygon(self.screen, fin2_points, self.COLOR_SHIP_FIN)
        pygame.gfxdraw.filled_polygon(self.screen, fin2_points, self.COLOR_SHIP_FIN)
        
        # Window
        pygame.gfxdraw.aacircle(self.screen, ship_x + 15, ship_y + 25, 8, self.COLOR_SHIP_WINDOW)
        pygame.gfxdraw.filled_circle(self.screen, ship_x + 15, ship_y + 25, 8, self.COLOR_SHIP_WINDOW)

    def _draw_obstacles(self):
        for obs in self.obstacles:
            rect = pygame.Rect(int(obs['x']), int(obs['y']), int(obs['width']), int(obs['height']))
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect)
            # Add a slight highlight for depth
            pygame.draw.line(self.screen, (255, 150, 150), rect.topleft, rect.topright, 2)

    def _draw_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / p['max_lifespan']))))
            color = (*p['color'], alpha)
            size = int(p['size'] * (p['lifespan'] / p['max_lifespan']))
            if size > 0:
                 # Create a temporary surface for alpha blending
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size, size), size)
                self.screen.blit(temp_surf, (int(p['x']) - size, int(p['y']) - size))

    def _draw_player(self):
        x, y = int(self.player_pos[0]), int(self.player_pos[1])
        s = self.PLAYER_SIZE
        
        # Points for an upward-pointing triangle
        p1 = (x, y - s)
        p2 = (x - s // 2, y)
        p3 = (x + s // 2, y)
        
        # Draw anti-aliased filled triangle
        pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)

    def _get_player_rect(self):
        # A simplified rectangle for collision, slightly smaller than visual
        s = self.PLAYER_SIZE
        return pygame.Rect(self.player_pos[0] - s // 4, self.player_pos[1] - s * 0.75, s // 2, s * 0.75)

    def _generate_obstacle(self):
        gap = self.np_random.integers(180, 300)
        width = self.np_random.integers(40, 80)
        height = self.np_random.integers(30, 100)
        
        self.next_obstacle_x += gap
        
        new_obs = {
            'id': self.obstacle_id_counter,
            'x': self.next_obstacle_x,
            'y': self.GROUND_Y - height,
            'width': width,
            'height': height
        }
        self.obstacles.append(new_obs)
        self.obstacle_id_counter += 1
        self.next_obstacle_x += width

    def _create_explosion(self, position):
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 8)
            self.particles.append({
                'x': position[0], 'y': position[1],
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'lifespan': self.np_random.integers(30, 60),
                'max_lifespan': 60,
                'size': self.np_random.integers(2, 5),
                'color': self.COLOR_PLAYER,
            })
            
    def _create_sparks(self, player_rect, obs_rect):
        # Find closest point on obstacle to player center
        px, py = player_rect.center
        ox, oy = obs_rect.center
        
        closest_x = max(obs_rect.left, min(px, obs_rect.right))
        closest_y = max(obs_rect.top, min(py, obs_rect.bottom))

        for _ in range(3):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                'x': closest_x, 'y': closest_y,
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'lifespan': self.np_random.integers(10, 20),
                'max_lifespan': 20,
                'size': self.np_random.integers(1, 3),
                'color': self.COLOR_PARTICLE,
            })

    def close(self):
        pygame.font.quit()
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
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Planet Escape")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Game loop
    while running:
        # Action defaults
        movement_action = 0  # no-op
        space_action = 0     # released
        shift_action = 0     # released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]:
            movement_action = 1
        elif keys[pygame.K_DOWN]:
            movement_action = 2
        elif keys[pygame.K_LEFT]:
            movement_action = 3
        elif keys[pygame.K_RIGHT]:
            movement_action = 4
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1
            
        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(100) # Run at 100 FPS to match step rate

    env.close()