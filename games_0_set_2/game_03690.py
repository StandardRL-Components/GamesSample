
# Generated: 2025-08-28T00:06:48.784071
# Source Brief: brief_03690.md
# Brief Index: 3690

        
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
        "Controls: ←→ to steer. Hold space to thrust upwards. Dodge the asteroids and survive for 10 turns."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a ship through a dense asteroid field. Use your thrusters to navigate and avoid a fatal collision. Survive for 10 turns to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 10  # A "turn" is one step
        
        # Physics constants
        self.SHIP_THRUST = 0.6
        self.SHIP_TURN_RATE = 0.4
        self.SHIP_DRAG = 0.98
        self.GRAVITY = 0.08
        self.MAX_VELOCITY = 6

        # Visual constants
        self.COLOR_BG = (10, 15, 25)
        self.COLOR_SHIP = (230, 240, 255)
        self.COLOR_SHIP_GLOW = (180, 200, 255)
        self.COLOR_ASTEROID = [(80, 85, 95), (90, 95, 105), (100, 105, 115)]
        self.COLOR_UI = (220, 220, 220)
        self.COLOR_PARTICLE = (255, 255, 255)
        
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
        try:
            self.font_ui = pygame.font.Font(None, 28)
            self.font_game_over = pygame.font.Font(None, 64)
        except IOError:
            self.font_ui = pygame.font.SysFont("sans-serif", 28)
            self.font_game_over = pygame.font.SysFont("sans-serif", 64)

        # Initialize state variables
        self.ship_pos = None
        self.ship_vel = None
        self.ship_poly = None
        self.asteroids = []
        self.thrust_particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.np_random = None
        
        # Initialize state variables
        self.reset()

        # Run validation
        # self.validate_implementation() # Commented out for submission, but useful for testing
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.ship_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 40)
        self.ship_vel = pygame.Vector2(0, 0)
        
        self.asteroids = []
        self.thrust_particles = []

        # Spawn initial asteroids to make the first step challenging
        for _ in range(10):
            self._spawn_asteroid(initial_spawn=True)

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        reward = 0
        terminated = False

        # Apply action penalty for no-op
        if movement == 0 and not space_held:
            reward -= 0.2

        # Update game logic
        self._update_ship(movement, space_held)
        self._update_asteroids()
        self._update_particles()
        self._spawn_asteroid()

        # Check for termination conditions
        if self._check_collision():
            # sfx: player_explosion.wav
            reward = -100.0
            terminated = True
            self.game_over = True
            self.win = False
        else:
            self.steps += 1
            self.score += 1 # Score is turns survived
            reward += 1.0 # Reward for surviving a step

            if self.steps >= self.MAX_STEPS:
                # sfx: victory_fanfare.wav
                reward = 100.0
                terminated = True
                self.game_over = True
                self.win = True

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_ship(self, movement, space_held):
        # Apply thrust
        if space_held:
            # sfx: thrust_loop.wav
            self.ship_vel.y -= self.SHIP_THRUST
            self._create_thrust_particles()
        
        # Apply turning
        if movement == 3: # Left
            self.ship_vel.x -= self.SHIP_TURN_RATE
        elif movement == 4: # Right
            self.ship_vel.x += self.SHIP_TURN_RATE

        # Apply physics
        self.ship_vel.y += self.GRAVITY
        self.ship_vel *= self.SHIP_DRAG
        
        # Clamp velocity
        self.ship_vel.x = max(-self.MAX_VELOCITY, min(self.MAX_VELOCITY, self.ship_vel.x))
        self.ship_vel.y = max(-self.MAX_VELOCITY, min(self.MAX_VELOCITY, self.ship_vel.y))

        self.ship_pos += self.ship_vel

        # Clamp position to screen bounds
        self.ship_pos.x = max(10, min(self.WIDTH - 10, self.ship_pos.x))
        self.ship_pos.y = max(10, min(self.HEIGHT - 10, self.ship_pos.y))

    def _update_asteroids(self):
        asteroids_to_keep = []
        for ast in self.asteroids:
            ast['pos'] += ast['vel']
            if -ast['radius'] < ast['pos'].x < self.WIDTH + ast['radius'] and \
               -ast['radius'] < ast['pos'].y < self.HEIGHT + ast['radius']:
                asteroids_to_keep.append(ast)
        self.asteroids = asteroids_to_keep

    def _spawn_asteroid(self, initial_spawn=False):
        num_to_spawn = 5 if not initial_spawn else 1
        for _ in range(num_to_spawn):
            radius = self.np_random.integers(10, 25)
            x = self.np_random.uniform(0, self.WIDTH)
            y = self.np_random.uniform(-self.HEIGHT, -radius) if initial_spawn else -radius
            
            angle_rad = math.radians(self.np_random.uniform(-15, 15))
            speed = self.np_random.uniform(2, 5)
            
            vel = pygame.Vector2(math.sin(angle_rad) * speed, math.cos(angle_rad) * speed)
            
            color = random.choice(self.COLOR_ASTEROID)
            
            self.asteroids.append({
                'pos': pygame.Vector2(x, y),
                'vel': vel,
                'radius': radius,
                'color': color
            })

    def _check_collision(self):
        ship_radius = 8 # Collision radius for ship
        for ast in self.asteroids:
            distance = self.ship_pos.distance_to(ast['pos'])
            if distance < ship_radius + ast['radius']:
                return True
        return False

    def _create_thrust_particles(self):
        for _ in range(3):
            particle_pos = self.ship_pos + pygame.Vector2(self.np_random.uniform(-3, 3), 10)
            particle_vel = pygame.Vector2(self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(2, 4))
            particle_life = self.np_random.integers(10, 20)
            self.thrust_particles.append({'pos': particle_pos, 'vel': particle_vel, 'life': particle_life})

    def _update_particles(self):
        particles_to_keep = []
        for p in self.thrust_particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] > 0:
                particles_to_keep.append(p)
        self.thrust_particles = particles_to_keep

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        self._render_particles()
        self._render_asteroids()
        if not (self.game_over and not self.win): # Don't render ship if crashed
            self._render_ship()

    def _render_ship(self):
        x, y = int(self.ship_pos.x), int(self.ship_pos.y)
        points = [
            (x, y - 12),
            (x - 8, y + 8),
            (x + 8, y + 8)
        ]
        # Glow effect
        pygame.gfxdraw.filled_trigon(self.screen, points[0][0], points[0][1],
                                     points[1][0], points[1][1],
                                     points[2][0], points[2][1], self.COLOR_SHIP_GLOW)
        pygame.gfxdraw.aatrigon(self.screen, points[0][0], points[0][1],
                                     points[1][0], points[1][1],
                                     points[2][0], points[2][1], self.COLOR_SHIP_GLOW)
        # Main body
        pygame.gfxdraw.filled_trigon(self.screen, points[0][0], points[0][1],
                                     points[1][0]+2, points[1][1]-2,
                                     points[2][0]-2, points[2][1]-2, self.COLOR_SHIP)

    def _render_asteroids(self):
        for ast in self.asteroids:
            x, y, r = int(ast['pos'].x), int(ast['pos'].y), int(ast['radius'])
            pygame.gfxdraw.filled_circle(self.screen, x, y, r, ast['color'])
            pygame.gfxdraw.aacircle(self.screen, x, y, r, tuple(c*0.8 for c in ast['color']))

    def _render_particles(self):
        for p in self.thrust_particles:
            life_ratio = p['life'] / 20.0
            size = int(3 * life_ratio)
            if size > 0:
                color_val = int(255 * life_ratio)
                color = (color_val, color_val, color_val)
                pygame.draw.circle(self.screen, color, (int(p['pos'].x), int(p['pos'].y)), size)

    def _render_ui(self):
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))
        
        turns_text = self.font_ui.render(f"Turn: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI)
        self.screen.blit(turns_text, (self.WIDTH - turns_text.get_width() - 10, 10))

        if self.game_over:
            if self.win:
                end_text = self.font_game_over.render("YOU WIN!", True, (100, 255, 100))
            else:
                end_text = self.font_game_over.render("GAME OVER", True, (255, 100, 100))
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "win": self.win,
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    
    # --- Manual Play ---
    # This part allows a human to play the game.
    # It requires a window, so we'll re-init pygame for display.
    pygame.display.init()
    pygame.display.set_caption("Asteroid Dodger")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    print(env.user_guide)

    while not done:
        # Action mapping for human play
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # In a real game loop, you'd call step once per frame/decision.
        # Here we add a delay to make it playable.
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        env.clock.tick(30) # Limit to 30 FPS for playability

    print(f"Game Over. Final Info: {info}")
    
    # Wait a bit before closing
    pygame.time.wait(2000)
    env.close()