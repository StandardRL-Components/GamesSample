import gymnasium as gym
import os
import pygame
import numpy as np
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:35:38.822830
# Source Brief: brief_00503.md
# Brief Index: 503
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls three rotating hexagons
    to absorb matching-sized squares. The goal is to absorb 100 squares
    before the hexagons grow too large and hit the screen edges, or time runs out.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Control three rotating hexagons to absorb matching-sized squares. Absorb 100 squares to win, "
        "but be careful not to let the hexagons grow too large and hit the walls."
    )
    user_guide = (
        "Use ↑↓ arrow keys to rotate the left hexagon and ←→ to rotate the middle one. "
        "Use Space and Shift to rotate the right hexagon. Match hexagon sizes with incoming squares to absorb them."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30  # Assumed FPS for game logic calculations

        # Colors
        self.COLOR_BG_TOP = (10, 0, 20)
        self.COLOR_BG_BOTTOM = (30, 0, 50)
        self.COLOR_HEX_1 = (0, 255, 255)  # Cyan
        self.COLOR_HEX_2 = (0, 192, 255)  # Light Blue
        self.COLOR_HEX_3 = (0, 128, 255)  # Blue
        self.COLOR_SQUARE = (255, 80, 80) # Red
        self.COLOR_PARTICLE = (255, 255, 150) # Yellow/White
        self.COLOR_TEXT = (220, 220, 220)

        # Game Parameters
        self.INITIAL_SPAWN_PROB = 0.03
        self.SPAWN_RATE_INCREASE_PER_SEC = 0.01
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 100
        self.ROTATION_SPEED = math.radians(5) # Radians per step
        
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
        self.font = pygame.font.Font(None, 36)
        
        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.hexagons = []
        self.squares = []
        self.particles = []
        self.current_spawn_prob = 0.0

        # self.reset() is called by the wrapper/runner, not needed here
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_spawn_prob = self.INITIAL_SPAWN_PROB

        # Initialize Hexagons
        # Format: {'pos': (x, y), 'size': radius, 'angle': rad, 'color': (r,g,b), 'prev_int_size': int}
        self.hexagons = [
            {'pos': (self.WIDTH * 0.25, self.HEIGHT * 0.5), 'size': 20, 'angle': 0, 'color': self.COLOR_HEX_1, 'prev_int_size': 20},
            {'pos': (self.WIDTH * 0.5, self.HEIGHT * 0.5), 'size': 35, 'angle': 0, 'color': self.COLOR_HEX_2, 'prev_int_size': 35},
            {'pos': (self.WIDTH * 0.75, self.HEIGHT * 0.5), 'size': 50, 'angle': 0, 'color': self.COLOR_HEX_3, 'prev_int_size': 50},
        ]

        self.squares = []
        self.particles = []
        
        # Spawn some initial squares to make the start less empty
        for _ in range(5):
            self._spawn_square()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # 1. Handle Input
        self._handle_input(action)

        # 2. Update Game State
        self._update_squares()
        reward += self._handle_collisions()
        self._update_particles()
        
        # Increase spawn rate over time
        self.current_spawn_prob += self.SPAWN_RATE_INCREASE_PER_SEC / self.FPS

        # 3. Check for Termination
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100 # Win bonus
            else:
                reward -= 100 # Loss penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action
        
        # Action mapping adaptation:
        # Movement[0] (up/down) -> Hexagon 1
        if movement == 1: # Up (interpreted as CW)
            self.hexagons[0]['angle'] += self.ROTATION_SPEED
        elif movement == 2: # Down (interpreted as CCW)
            self.hexagons[0]['angle'] -= self.ROTATION_SPEED
            
        # Movement[0] (left/right) -> Hexagon 2
        if movement == 3: # Left (interpreted as CCW)
            self.hexagons[1]['angle'] -= self.ROTATION_SPEED
        elif movement == 4: # Right (interpreted as CW)
            self.hexagons[1]['angle'] += self.ROTATION_SPEED

        # space/shift -> Hexagon 3
        if space_held == 1: # Space for CW
            self.hexagons[2]['angle'] += self.ROTATION_SPEED
        if shift_held == 1: # Shift for CCW
            self.hexagons[2]['angle'] -= self.ROTATION_SPEED

    def _update_squares(self):
        # Move existing squares
        for sq in self.squares[:]:
            sq['pos'] = (sq['pos'][0] + sq['vel'][0], sq['pos'][1] + sq['vel'][1])
            # Remove if off-screen
            if not (-sq['size'] < sq['pos'][0] < self.WIDTH + sq['size'] and \
                    -sq['size'] < sq['pos'][1] < self.HEIGHT + sq['size']):
                self.squares.remove(sq)
        
        # Spawn new squares
        if self.np_random.random() < self.current_spawn_prob:
            self._spawn_square()

    def _spawn_square(self):
        size = self.np_random.uniform(10, 40)
        edge = self.np_random.choice(['top', 'bottom', 'left', 'right'])
        
        if edge == 'top':
            pos = (self.np_random.uniform(0, self.WIDTH), -size)
            vel = (self.np_random.uniform(-1, 1), self.np_random.uniform(1, 3))
        elif edge == 'bottom':
            pos = (self.np_random.uniform(0, self.WIDTH), self.HEIGHT + size)
            vel = (self.np_random.uniform(-1, 1), self.np_random.uniform(-3, -1))
        elif edge == 'left':
            pos = (-size, self.np_random.uniform(0, self.HEIGHT))
            vel = (self.np_random.uniform(1, 3), self.np_random.uniform(-1, 1))
        else: # right
            pos = (self.WIDTH + size, self.np_random.uniform(0, self.HEIGHT))
            vel = (self.np_random.uniform(-3, -1), self.np_random.uniform(-1, 1))
            
        # Speed inversely proportional to size
        speed_modifier = 30 / max(10, size)
        vel = (vel[0] * speed_modifier, vel[1] * speed_modifier)

        self.squares.append({'pos': pos, 'size': size, 'vel': vel})

    def _handle_collisions(self):
        step_reward = 0
        for sq in self.squares[:]:
            for h in self.hexagons:
                dist = math.hypot(sq['pos'][0] - h['pos'][0], sq['pos'][1] - h['pos'][1])
                
                # Check for collision and size match
                size_match_lower = h['size'] * 0.7
                size_match_upper = h['size'] * 1.3
                if dist < h['size'] and size_match_lower < sq['size'] < size_match_upper:
                    # Successful absorption
                    self.squares.remove(sq)
                    self.score += 1
                    step_reward += 0.1 # Small reward for absorption
                    
                    # Grow hexagon
                    growth = max(1, sq['size'] * 0.1)
                    h['size'] += growth
                    
                    # Check for size-up reward
                    if int(h['size']) > h['prev_int_size']:
                        step_reward += 1.0
                        h['prev_int_size'] = int(h['size'])

                    # Create particles
                    self._create_particles(h['pos'], 20)
                    
                    # Sound effect placeholder
                    # pygame.mixer.Sound("absorb.wav").play()
                    
                    break # Square is absorbed, no need to check other hexagons
        return step_reward
        
    def _create_particles(self, pos, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            size = self.np_random.uniform(2, 5)
            lifetime = self.np_random.integers(15, 30) # in frames
            self.particles.append({'pos': list(pos), 'vel': vel, 'size': size, 'lifetime': lifetime})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifetime'] -= 1
            p['size'] *= 0.95 # Shrink
            if p['lifetime'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            return True
        
        for h in self.hexagons:
            if (h['pos'][0] - h['size'] < 0 or
                h['pos'][0] + h['size'] > self.WIDTH or
                h['pos'][1] - h['size'] < 0 or
                h['pos'][1] + h['size'] > self.HEIGHT):
                return True
        return False

    def _get_observation(self):
        # --- Render all game elements to self.screen ---
        self._render_background()
        self._render_particles()
        self._render_squares()
        self._render_hexagons()
        self._render_ui()
        
        # Convert to numpy array (H, W, C)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Simple gradient
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_hexagons(self):
        for h in self.hexagons:
            self._draw_hexagon_glow(self.screen, h['color'], h['pos'], h['size'])
            self._draw_hexagon(self.screen, h['color'], h['pos'], h['size'], h['angle'])
    
    def _draw_hexagon(self, surface, color, center, radius, angle):
        points = []
        for i in range(6):
            a = angle + (math.pi / 3) * i
            x = center[0] + radius * math.cos(a)
            y = center[1] + radius * math.sin(a)
            points.append((int(x), int(y)))
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)
    
    def _draw_hexagon_glow(self, surface, color, center, radius):
        # Draw multiple transparent circles to simulate a glow
        glow_color = (color[0], color[1], color[2], 20)
        for i in range(5):
            r = radius + i * 3
            # Need a temporary surface for alpha blending
            temp_surf = pygame.Surface((int(r*2), int(r*2)), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, glow_color, (int(r), int(r)), int(r))
            surface.blit(temp_surf, (int(center[0] - r), int(center[1] - r)))

    def _render_squares(self):
        for sq in self.squares:
            rect = pygame.Rect(int(sq['pos'][0] - sq['size']/2), int(sq['pos'][1] - sq['size']/2), int(sq['size']), int(sq['size']))
            pygame.draw.rect(self.screen, self.COLOR_SQUARE, rect)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / 30)) # Fade out
            color = (*self.COLOR_PARTICLE, alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            size = max(0, int(p['size']))
            
            # Use a temporary surface for alpha blending
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, (pos[0] - size, pos[1] - size))

    def _render_ui(self):
        score_text = self.font.render(f"Score: {self.score} / {self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.font.render(f"Steps: {self.steps} / {self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            status_text = "YOU WON!" if self.score >= self.WIN_SCORE else "GAME OVER"
            status_render = self.font.render(status_text, True, (255, 255, 255))
            text_rect = status_render.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(status_render, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "hex_sizes": [h['size'] for h in self.hexagons],
            "num_squares": len(self.squares),
            "spawn_prob": self.current_spawn_prob,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()


# Example of how to run the environment for human play
if __name__ == '__main__':
    # This block is for human play and debugging, not used by the evaluation system.
    # It requires a display.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame window for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Hexagon Absorber")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # --- Action Mapping for Human ---
        movement = 0 
        space = 0
        shift = 0

        keys = pygame.key.get_pressed()
        # Arrow keys for Hex 1 & 2
        if keys[pygame.K_UP]: movement = 1 # CW
        elif keys[pygame.K_DOWN]: movement = 2 # CCW
        elif keys[pygame.K_LEFT]: movement = 3 # CCW
        elif keys[pygame.K_RIGHT]: movement = 4 # CW
        
        # Space/Shift for Hex 3
        if keys[pygame.K_SPACE]: space = 1 # CW
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1 # CCW

        action = [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                if event.key == pygame.K_ESCAPE:
                    running = False

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"Game Over. Score: {info['score']}")
            # Optional: auto-reset
            # obs, info = env.reset()

        # --- Rendering ---
        # The observation is the frame, so we just need to display it
        frame = np.transpose(obs, (1, 0, 2)) # Transpose back for pygame display
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)

    env.close()