import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:22:51.449134
# Source Brief: brief_00394.md
# Brief Index: 394
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a shape-shifting entity.
    The goal is to survive, collect points by staying in different forms,
    and avoid obstacles.

    - Player can be a slow, safe SQUARE or a fast, risky CIRCLE.
    - Obstacles are red rectangles moving in sinusoidal patterns.
    - Victory: Reach 100 points.
    - Failure: Collide as a circle, or run out of time.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Control a shape-shifting entity to survive and score points. Switch between a slow, "
        "safe square and a fast, risky circle to avoid obstacles."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move. Press Shift to switch between square and circle forms."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 900  # 30 seconds at 30 FPS
    SCORE_GOAL = 100
    TIME_LIMIT_SECONDS = 30

    # Colors
    COLOR_BG = (15, 18, 33)
    COLOR_SQUARE = (60, 160, 255)
    COLOR_SQUARE_GLOW = (40, 80, 180)
    COLOR_CIRCLE = (255, 220, 50)
    COLOR_CIRCLE_GLOW = (200, 150, 40)
    COLOR_OBSTACLE = (255, 50, 80)
    COLOR_OBSTACLE_GLOW = (180, 40, 60)
    COLOR_UI_TEXT = (230, 230, 230)
    
    # Player settings
    PLAYER_SIZE = 20
    PLAYER_SPEED_SQUARE = 4.0
    PLAYER_SPEED_CIRCLE = 8.0
    
    # Game mechanics
    POINTS_PER_SEC_SQUARE = 2
    POINTS_PER_SEC_CIRCLE = 10
    SQUARE_COLLISION_PENALTY = 10
    OBSTACLE_COUNT = 8
    OBSTACLE_BASE_SPEED = 1.5
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        # --- Game State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.time_remaining = 0.0
        self.game_over = False
        
        self.player_pos = None
        self.player_shape = None # 'square' or 'circle'
        self.obstacles = []
        self.particles = []

        # Control flags
        self.shift_pressed_last_frame = False
        self.collision_cooldown = 0
        
        # This will be called in __init__ which will properly set up the initial state
        # self.reset() is called by the environment wrapper, so we don't need to call it here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Initialize Game State ---
        self.steps = 0
        self.score = 0.0
        self.time_remaining = self.TIME_LIMIT_SECONDS
        self.game_over = False
        
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.player_shape = 'square'
        
        self.obstacles.clear()
        self.particles.clear()
        for _ in range(self.OBSTACLE_COUNT):
            self._spawn_obstacle()

        # Reset control flags
        self.shift_pressed_last_frame = True # Prevent switching on first frame
        self.collision_cooldown = 0

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0.0
        
        # --- Update Game Logic ---
        self._handle_input(movement, shift_held)
        self._update_player()
        self._update_obstacles()
        self._update_particles()
        
        # --- Update Timers and Score ---
        self.steps += 1
        time_delta = 1 / self.FPS
        self.time_remaining -= time_delta
        if self.collision_cooldown > 0:
            self.collision_cooldown -= 1
        
        # Continuous points/reward based on shape
        if self.player_shape == 'square':
            self.score += self.POINTS_PER_SEC_SQUARE * time_delta
            reward += 0.02
        else: # circle
            self.score += self.POINTS_PER_SEC_CIRCLE * time_delta
            reward += 0.1
            # Spawn trail particles for circle
            self._spawn_particles(self.player_pos, self.COLOR_CIRCLE, 1, 1.0)
            
        # --- Collision Detection ---
        player_rect = pygame.Rect(
            self.player_pos.x - self.PLAYER_SIZE / 2,
            self.player_pos.y - self.PLAYER_SIZE / 2,
            self.PLAYER_SIZE, self.PLAYER_SIZE
        )
        
        for obstacle in self.obstacles:
            if player_rect.colliderect(obstacle['rect']):
                if self.player_shape == 'circle':
                    # Game over for circle
                    self.game_over = True
                    # Sound: Player_Explode.wav
                    self._spawn_particles(self.player_pos, self.COLOR_CIRCLE, 50, 3.0, 10.0)
                    break 
                elif self.player_shape == 'square' and self.collision_cooldown == 0:
                    # Penalty for square
                    self.score = max(0, self.score - self.SQUARE_COLLISION_PENALTY)
                    reward -= 10
                    self.collision_cooldown = self.FPS // 2 # 0.5 sec invulnerability
                    # Sound: Player_Hit.wav
                    self._spawn_particles(self.player_pos, self.COLOR_OBSTACLE, 20, 2.0, 5.0)
                    break
        
        # --- Termination Conditions ---
        terminated = self.game_over
        if self.score >= self.SCORE_GOAL:
            reward += 100 # Goal achievement reward
            terminated = True
        if self.time_remaining <= 0 or self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement, shift_held):
        # --- Player Movement ---
        velocity = pygame.Vector2(0, 0)
        speed = self.PLAYER_SPEED_CIRCLE if self.player_shape == 'circle' else self.PLAYER_SPEED_SQUARE
        
        if movement == 1: velocity.y = -1 # Up
        elif movement == 2: velocity.y = 1 # Down
        elif movement == 3: velocity.x = -1 # Left
        elif movement == 4: velocity.x = 1 # Right
        
        if velocity.length() > 0:
            velocity.normalize_ip()
            self.player_pos += velocity * speed

        # --- Shape Switching ---
        if shift_held and not self.shift_pressed_last_frame:
            self.player_shape = 'circle' if self.player_shape == 'square' else 'square'
            # Sound: Shape_Switch.wav
            color = self.COLOR_CIRCLE if self.player_shape == 'circle' else self.COLOR_SQUARE
            self._spawn_particles(self.player_pos, color, 30, 2.0, 4.0)

        self.shift_pressed_last_frame = shift_held

    def _update_player(self):
        # Clamp player position to screen bounds
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE/2, self.SCREEN_WIDTH - self.PLAYER_SIZE/2)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_SIZE/2, self.SCREEN_HEIGHT - self.PLAYER_SIZE/2)

    def _update_obstacles(self):
        # Difficulty scaling: speed increases over time
        time_elapsed_ratio = (self.TIME_LIMIT_SECONDS - self.time_remaining) / self.TIME_LIMIT_SECONDS
        speed_multiplier = 1.0 + time_elapsed_ratio * 1.5 # Up to 2.5x speed at the end

        for obs in self.obstacles:
            # Sinusoidal movement
            if obs['axis'] == 'x': # Moves horizontally
                obs['pos'].x += obs['speed'] * speed_multiplier * obs['direction']
                offset = obs['amplitude'] * math.sin(self.steps * obs['frequency'] + obs['phase'])
                obs['pos'].y = obs['start_pos'].y + offset
                if obs['pos'].x > self.SCREEN_WIDTH + obs['size'].x: obs['pos'].x = -obs['size'].x
                if obs['pos'].x < -obs['size'].x: obs['pos'].x = self.SCREEN_WIDTH + obs['size'].x
            else: # Moves vertically
                obs['pos'].y += obs['speed'] * speed_multiplier * obs['direction']
                offset = obs['amplitude'] * math.sin(self.steps * obs['frequency'] + obs['phase'])
                obs['pos'].x = obs['start_pos'].x + offset
                if obs['pos'].y > self.SCREEN_HEIGHT + obs['size'].y: obs['pos'].y = -obs['size'].y
                if obs['pos'].y < -obs['size'].y: obs['pos'].y = self.SCREEN_HEIGHT + obs['size'].y
            
            obs['rect'].topleft = obs['pos']

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['radius'] -= p['decay']
            if p['lifetime'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Render Particles ---
        for p in self.particles:
            if p['radius'] > 1:
                pygame.draw.circle(self.screen, p['color'], p['pos'], max(0, int(p['radius'])))

        # --- Render Obstacles ---
        for obs in self.obstacles:
            # Glow effect
            glow_rect = obs['rect'].inflate(8, 8)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_GLOW, glow_rect, border_radius=5)
            # Main shape
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs['rect'], border_radius=3)
            
        # --- Render Player ---
        if not self.game_over:
            pos = (int(self.player_pos.x), int(self.player_pos.y))
            size = self.PLAYER_SIZE
            glow_size = int(size * 2.5)
            
            if self.player_shape == 'square':
                color, glow_color = self.COLOR_SQUARE, self.COLOR_SQUARE_GLOW
                # Glow
                glow_surface = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
                pygame.draw.rect(glow_surface, glow_color + (60,), glow_surface.get_rect(), border_radius=8)
                self.screen.blit(glow_surface, (pos[0] - glow_size//2, pos[1] - glow_size//2))
                # Main shape
                player_rect = pygame.Rect(pos[0] - size//2, pos[1] - size//2, size, size)
                pygame.draw.rect(self.screen, color, player_rect, border_radius=3)
            else: # circle
                color, glow_color = self.COLOR_CIRCLE, self.COLOR_CIRCLE_GLOW
                # Glow using gfxdraw for antialiasing
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(size * 0.75), glow_color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(size * 0.75), glow_color)
                # Main shape
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(size * 0.5), color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(size * 0.5), color)

    def _render_ui(self):
        # --- Score Display ---
        score_text = self.font_large.render(f"{int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))

        # --- Timer Display ---
        timer_text = self.font_large.render(f"{max(0, self.time_remaining):.1f}", True, self.COLOR_UI_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(timer_text, timer_rect)
        
        # --- Shape Display ---
        shape_text = self.font_medium.render(self.player_shape.upper(), True, self.COLOR_UI_TEXT)
        shape_rect = shape_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 30))
        self.screen.blit(shape_text, shape_rect)

        # --- Game Over Message ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "GOAL REACHED!" if self.score >= self.SCORE_GOAL else "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, self.COLOR_UI_TEXT)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20))
            self.screen.blit(end_text, end_rect)
            
            final_score_text = self.font_medium.render(f"Final Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
            final_score_rect = final_score_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 20))
            self.screen.blit(final_score_text, final_score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "player_shape": self.player_shape
        }

    def _spawn_obstacle(self):
        axis = self.np_random.choice(['x', 'y'])
        size = pygame.Vector2(self.np_random.integers(20, 60), self.np_random.integers(20, 60))
        
        if axis == 'x': # Horizontal movement
            start_pos = pygame.Vector2(self.np_random.uniform(-size.x, self.SCREEN_WIDTH + size.x),
                                      self.np_random.uniform(size.y, self.SCREEN_HEIGHT - size.y))
        else: # Vertical movement
            start_pos = pygame.Vector2(self.np_random.uniform(size.x, self.SCREEN_WIDTH - size.x),
                                      self.np_random.uniform(-size.y, self.SCREEN_HEIGHT + size.y))

        self.obstacles.append({
            'pos': start_pos.copy(),
            'start_pos': start_pos.copy(),
            'size': size,
            'rect': pygame.Rect(start_pos, size),
            'speed': self.OBSTACLE_BASE_SPEED * self.np_random.uniform(0.8, 1.5),
            'direction': self.np_random.choice([-1, 1]),
            'axis': axis,
            'amplitude': self.np_random.uniform(20, 100),
            'frequency': self.np_random.uniform(0.01, 0.05),
            'phase': self.np_random.uniform(0, 2 * math.pi)
        })

    def _spawn_particles(self, pos, color, count, lifetime_mult=1.0, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifetime = self.np_random.integers(10, 30) * lifetime_mult
            radius = self.np_random.uniform(2, 6)
            self.particles.append({
                'pos': pos.copy() + vel * self.np_random.uniform(1, 5), # Start away from center
                'vel': vel,
                'lifetime': lifetime,
                'radius': radius,
                'decay': radius / lifetime,
                'color': color
            })
            
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        obs, _ = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To use, you might need to remove the SDL_VIDEODRIVER dummy setting
    # and install pygame: pip install pygame
    # For example, comment out line 3: # os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    
    # Re-enable video driver for manual play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Shape Shifter Survival")
    clock = pygame.time.Clock()
    
    terminated = False
    
    while not terminated:
        # --- Manual Control ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Frame Rate Control ---
        clock.tick(env.metadata["render_fps"])

    print(f"Game Over. Final Info: {info}")
    env.close()