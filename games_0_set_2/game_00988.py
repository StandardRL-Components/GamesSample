
# Generated: 2025-08-27T15:25:33.632567
# Source Brief: brief_00988.md
# Brief Index: 988

        
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

    # User-facing control string
    user_guide = (
        "Controls: ↑↓←→ to move. Avoid the red lasers and reach the silver spaceship."
    )

    # User-facing game description
    game_description = (
        "Guide an alien through laser grids to reach its spaceship in a fast-paced, grid-based puzzle game."
    )

    # Frames advance on action, not automatically
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 20, 15
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (10, 10, 20)
    COLOR_GRID = (30, 60, 90)
    COLOR_ALIEN = (50, 255, 50)
    COLOR_ALIEN_GLOW = (50, 255, 50, 50)
    COLOR_SPACESHIP = (192, 192, 210)
    COLOR_SPACESHIP_GLOW = (192, 192, 210, 30)
    COLOR_LASER_CORE = (255, 20, 20)
    COLOR_LASER_GLOW = (255, 20, 20, 40)
    COLOR_TEXT = (220, 220, 220)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Grid and rendering properties
        self.cell_width = (self.SCREEN_WIDTH - 40) / self.GRID_WIDTH
        self.cell_height = (self.SCREEN_HEIGHT - 50) / self.GRID_HEIGHT
        self.x_offset = 20
        self.y_offset = 40
        
        # Initialize state variables
        self.alien_pos = [0, 0]
        self.spaceship_pos = [0, 0]
        self.lasers = []
        self.laser_speed = 0.0
        self.particles = []
        self.steps = 0
        self.score = 0.0
        self.game_over_message = ""
        
        self.reset()
        
        # This check is for development, comment out for production
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0.0
        self.game_over_message = ""
        
        # Place alien and spaceship
        self.alien_pos = [1, self.GRID_HEIGHT // 2]
        self.spaceship_pos = [self.GRID_WIDTH - 2, self.GRID_HEIGHT // 2]

        # Initialize lasers
        self.lasers = []
        self.laser_speed = 0.5
        
        # Vertical Lasers
        for _ in range(4):
            while True:
                pos = self.rng.integers(0, self.GRID_WIDTH)
                if pos != self.alien_pos[0] and pos != self.spaceship_pos[0]:
                    break
            direction = self.rng.choice([-1, 1])
            self.lasers.append({'type': 'v', 'pos': float(pos), 'dir': direction})
        
        # Horizontal Lasers
        for _ in range(4):
            while True:
                pos = self.rng.integers(0, self.GRID_HEIGHT)
                if pos != self.alien_pos[1] and pos != self.spaceship_pos[1]:
                    break
            direction = self.rng.choice([-1, 1])
            self.lasers.append({'type': 'h', 'pos': float(pos), 'dir': direction})

        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        
        reward = -0.1  # Cost of existing
        terminated = False

        if not self.game_over_message:
            # --- Game Logic ---
            self._move_alien(movement)
            self._update_lasers()
            self._update_particles()
            
            # --- Check Conditions ---
            if self.alien_pos == self.spaceship_pos:
                reward = 10.0
                terminated = True
                self.game_over_message = "YOU ESCAPED!"
                # sfx: win_sound

            if self._check_collisions():
                reward = -10.0
                terminated = True
                self._create_explosion(self.alien_pos)
                self.game_over_message = "GAME OVER"
                # sfx: explosion_sound
        
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            self.game_over_message = "OUT OF TIME"

        self.steps += 1
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _move_alien(self, movement):
        # sfx: move_blip
        x, y = self.alien_pos
        if movement == 1: y -= 1  # Up
        elif movement == 2: y += 1  # Down
        elif movement == 3: x -= 1  # Left
        elif movement == 4: x += 1  # Right
        
        # Clamp to grid boundaries
        self.alien_pos[0] = max(0, min(self.GRID_WIDTH - 1, x))
        self.alien_pos[1] = max(0, min(self.GRID_HEIGHT - 1, y))

    def _update_lasers(self):
        # Difficulty scaling
        if self.steps > 0 and self.steps % 500 == 0:
            self.laser_speed = min(2.0, self.laser_speed + 0.05)

        for laser in self.lasers:
            laser['pos'] += laser['dir'] * self.laser_speed
            
            # Bounce off edges
            if laser['type'] == 'v' and not (0 <= laser['pos'] < self.GRID_WIDTH):
                laser['dir'] *= -1
                laser['pos'] = np.clip(laser['pos'], 0, self.GRID_WIDTH - 0.01)
            elif laser['type'] == 'h' and not (0 <= laser['pos'] < self.GRID_HEIGHT):
                laser['dir'] *= -1
                laser['pos'] = np.clip(laser['pos'], 0, self.GRID_HEIGHT - 0.01)

    def _check_collisions(self):
        ax, ay = self.alien_pos
        for laser in self.lasers:
            if laser['type'] == 'v' and int(laser['pos']) == ax:
                return True
            if laser['type'] == 'h' and int(laser['pos']) == ay:
                return True
        return False

    def _world_to_screen(self, x, y):
        px = self.x_offset + x * self.cell_width
        py = self.y_offset + y * self.cell_height
        return int(px), int(py)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_WIDTH + 1):
            x = self.x_offset + i * self.cell_width
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.y_offset), (x, self.y_offset + self.GRID_HEIGHT * self.cell_height))
        for i in range(self.GRID_HEIGHT + 1):
            y = self.y_offset + i * self.cell_height
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.x_offset, y), (self.x_offset + self.GRID_WIDTH * self.cell_width, y))

        # Draw spaceship
        self._render_spaceship()

        # Draw lasers
        self._render_lasers()

        # Draw alien if not destroyed
        if not self.game_over_message or self.game_over_message == "YOU ESCAPED!":
             self._render_alien()

        # Draw particles
        self._render_particles()

    def _render_spaceship(self):
        sx, sy = self._world_to_screen(self.spaceship_pos[0], self.spaceship_pos[1])
        cx = sx + self.cell_width / 2
        cy = sy + self.cell_height / 2
        radius = int(min(self.cell_width, self.cell_height) * 0.4)
        
        # Pulsing glow
        glow_radius = radius + 5 + int(3 * math.sin(self.steps * 0.1))
        pygame.gfxdraw.filled_circle(self.screen, int(cx), int(cy), glow_radius, self.COLOR_SPACESHIP_GLOW)
        
        # Main body
        pygame.gfxdraw.aacircle(self.screen, int(cx), int(cy), radius, self.COLOR_SPACESHIP)
        pygame.gfxdraw.filled_circle(self.screen, int(cx), int(cy), radius, self.COLOR_SPACESHIP)
        
        # Cockpit
        pygame.gfxdraw.filled_circle(self.screen, int(cx), int(cy), radius // 2, self.COLOR_BG)

    def _render_lasers(self):
        for laser in self.lasers:
            if laser['type'] == 'v':
                lx, _ = self._world_to_screen(laser['pos'], 0)
                lx += self.cell_width / 2
                start_y = self.y_offset
                end_y = self.y_offset + self.GRID_HEIGHT * self.cell_height
                
                # Glow
                pygame.draw.line(self.screen, self.COLOR_LASER_GLOW, (lx, start_y), (lx, end_y), 7)
                # Core
                pygame.draw.line(self.screen, self.COLOR_LASER_CORE, (lx, start_y), (lx, end_y), 3)

            elif laser['type'] == 'h':
                _, ly = self._world_to_screen(0, laser['pos'])
                ly += self.cell_height / 2
                start_x = self.x_offset
                end_x = self.x_offset + self.GRID_WIDTH * self.cell_width
                
                # Glow
                pygame.draw.line(self.screen, self.COLOR_LASER_GLOW, (start_x, ly), (end_x, ly), 7)
                # Core
                pygame.draw.line(self.screen, self.COLOR_LASER_CORE, (start_x, ly), (end_x, ly), 3)

    def _render_alien(self):
        ax, ay = self._world_to_screen(self.alien_pos[0], self.alien_pos[1])
        cx = ax + self.cell_width / 2
        cy = ay + self.cell_height / 2
        radius = int(min(self.cell_width, self.cell_height) * 0.35)

        # Glow
        pygame.gfxdraw.filled_circle(self.screen, int(cx), int(cy), radius + 5, self.COLOR_ALIEN_GLOW)
        
        # Body
        pygame.gfxdraw.aacircle(self.screen, int(cx), int(cy), radius, self.COLOR_ALIEN)
        pygame.gfxdraw.filled_circle(self.screen, int(cx), int(cy), radius, self.COLOR_ALIEN)

    def _render_ui(self):
        # Top UI panel background
        pygame.draw.rect(self.screen, (0,0,0), (0, 0, self.SCREEN_WIDTH, self.y_offset - 1))

        # Render step count
        step_text = self.font_small.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(step_text, (10, 10))

        # Render score
        score_text = self.font_small.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Render game over message
        if self.game_over_message:
            msg_surf = self.font_large.render(self.game_over_message, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _create_explosion(self, grid_pos):
        sx, sy = self._world_to_screen(grid_pos[0], grid_pos[1])
        center_x = sx + self.cell_width / 2
        center_y = sy + self.cell_height / 2
        
        for _ in range(30):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * 3 + 1
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.rng.integers(20, 40)
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': vel,
                'lifespan': lifespan,
                'max_life': lifespan,
                'color': self.COLOR_ALIEN
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_life']))
            color = (*p['color'][:3], alpha)
            radius = int(3 * (p['lifespan'] / p['max_life']))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, color)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    # Override metadata for human rendering
    env.metadata["render_modes"] = ["human"]
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Laser Grid Alien")

    terminated = False
    action = env.action_space.sample()
    action[0] = 0 # Start with no-op

    print(env.game_description)
    print(env.user_guide)

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            
            # Reset action on key up
            if event.type == pygame.KEYUP:
                action[0] = 0 # No movement

        # --- Keypress handling ---
        # Since auto_advance is False, we only step on keypresses
        # But for human play, it's better to poll held keys and step continuously
        keys = pygame.key.get_pressed()
        
        # Default to no-op if no movement key is pressed
        movement_action = 0 
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4

        # Only step if a key is pressed or to advance the game state
        action[0] = movement_action
        
        obs, reward, terminated, truncated, info = env.step(action)

        # Draw the observation to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(10) # Control game speed for human play

    print(f"Game Over! Final Score: {info['score']:.1f}, Steps: {info['steps']}")
    env.close()