import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:07:36.987846
# Source Brief: brief_00277.md
# Brief Index: 277
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A stealth-puzzle Gymnasium environment where a geometric ghost teleports
    between shapes, terraforming them to blend in with patrolling Scanners
    and reach the anomaly source.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A stealth-puzzle game where you teleport a ghost between shapes, changing their color to avoid detection by patrolling scanners and reach the final anomaly."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to teleport between shapes. Press space to change your shape's color to match the nearest one, and shift to match the anomaly's color."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000
    TARGET_FPS = 30

    # Colors
    COLOR_BG_TOP = (10, 20, 30)
    COLOR_BG_BOTTOM = (5, 10, 15)
    COLOR_ANOMALY = (0, 150, 255)
    COLOR_SCANNER = (255, 0, 50)
    COLOR_GHOST = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    SHAPE_PALETTE = [
        (255, 80, 200),   # Magenta
        (80, 255, 200),   # Cyan
        (255, 200, 80),   # Yellow
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Game state variables are initialized in reset()
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.shapes = []
        self.scanners = []
        self.anomaly = {}
        self.player_shape_idx = 0
        self.last_dist_to_anomaly = 0.0
        self.particles = []
        self.anomaly_pulse = 0.0
        self.scanner_base_speed = 0.05
        
        # self.reset() is called by the wrapper/user, not needed here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.particles = []
        
        self._generate_level()
        
        player_pos = self.shapes[self.player_shape_idx]['pos']
        self.last_dist_to_anomaly = self._get_distance(player_pos, self.anomaly['pos'])
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # Unpack factorized action
        movement_action = action[0]
        terraform_nearest_action = action[1] == 1
        terraform_anomaly_action = action[2] == 1

        reward = 0.0
        terminated = False
        truncated = False
        
        # 1. Handle Player Actions
        terraform_reward = self._handle_terraform(terraform_nearest_action, terraform_anomaly_action)
        teleport_success = self._handle_teleport(movement_action)
        reward += terraform_reward

        # 2. Update World State
        self._update_scanners()
        self._update_particles()
        self.anomaly_pulse = (self.anomaly_pulse + 0.1) % (2 * math.pi)

        # 3. Calculate Rewards & Check Termination
        player_pos = self.shapes[self.player_shape_idx]['pos']
        current_dist = self._get_distance(player_pos, self.anomaly['pos'])
        
        # Distance-based reward
        dist_reward = (self.last_dist_to_anomaly - current_dist) * 0.1
        reward += dist_reward
        self.last_dist_to_anomaly = current_dist

        # Win condition
        if current_dist < self.anomaly['radius'] + self.shapes[self.player_shape_idx]['size']:
            # Sound: Victory_fanfare.wav
            reward += 100.0
            self.game_over = True
            terminated = True
            self.win_message = "ANOMALY REACHED"
        
        # Loss condition (detection)
        if not terminated and self._check_detection():
            # Sound: Detection_alarm.wav
            reward -= 100.0
            self.game_over = True
            terminated = True
            self.win_message = "DETECTED"

        # Max steps truncation
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            truncated = True
        
        self.score += reward

        # The game is over if it's terminated or truncated
        if terminated or truncated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _generate_level(self):
        """Procedurally generates the positions and properties of all game entities."""
        self.shapes = []
        self.scanners = []
        
        # Place anomaly on the right side
        self.anomaly = {
            'pos': (self.np_random.integers(self.SCREEN_WIDTH * 0.8, self.SCREEN_WIDTH - 30),
                    self.np_random.integers(30, self.SCREEN_HEIGHT - 30)),
            'radius': 15,
            'color': self.COLOR_ANOMALY
        }

        # Generate shapes
        num_shapes = self.np_random.integers(15, 25)
        for _ in range(num_shapes):
            placed = False
            while not placed:
                pos = (self.np_random.integers(30, self.SCREEN_WIDTH - 30),
                       self.np_random.integers(30, self.SCREEN_HEIGHT - 30))
                size = self.np_random.integers(10, 16)
                
                # Ensure no overlap with other shapes
                if all(self._get_distance(pos, s['pos']) > s['size'] + size + 20 for s in self.shapes):
                    shape_type = self.np_random.choice(['circle', 'square', 'triangle'])
                    color_idx = self.np_random.integers(0, len(self.SHAPE_PALETTE))
                    self.shapes.append({'pos': pos, 'type': shape_type, 'size': size, 'color_idx': color_idx})
                    placed = True
        
        # Select starting shape for player (farthest from anomaly)
        self.player_shape_idx = max(range(len(self.shapes)), 
                                    key=lambda i: self._get_distance(self.shapes[i]['pos'], self.anomaly['pos']))

        # Generate scanners
        num_scanners = self.np_random.integers(2, 4)
        for _ in range(num_scanners):
            pos = (self.np_random.integers(100, self.SCREEN_WIDTH - 100),
                   self.np_random.integers(50, self.SCREEN_HEIGHT - 50))
            color_idx = self.np_random.integers(0, len(self.SHAPE_PALETTE))
            self.scanners.append({
                'pos': pos,
                'angle': self.np_random.uniform(0, 2 * math.pi),
                'length': self.np_random.integers(80, 120),
                'detection_radius': 150,
                'color_idx': color_idx
            })

    def _handle_teleport(self, movement_action):
        """Finds the best target shape based on direction and teleports the player."""
        if movement_action == 0:
            return False

        current_pos = self.shapes[self.player_shape_idx]['pos']
        targets = []
        
        direction_vectors = {
            1: (0, -1),  # Up
            2: (0, 1),   # Down
            3: (-1, 0),  # Left
            4: (1, 0),   # Right
        }
        target_vector = direction_vectors[movement_action]

        for i, shape in enumerate(self.shapes):
            if i == self.player_shape_idx:
                continue
            
            shape_vec = (shape['pos'][0] - current_pos[0], shape['pos'][1] - current_pos[1])
            dist = self._get_distance(current_pos, shape['pos'])
            if dist == 0: continue

            norm_shape_vec = (shape_vec[0] / dist, shape_vec[1] / dist)
            
            # Dot product to check if it's in the right general direction
            dot_product = norm_shape_vec[0] * target_vector[0] + norm_shape_vec[1] * target_vector[1]
            
            if dot_product > 0.5: # Must be within roughly 60 degrees of the target direction
                # Score prioritizes alignment, then closeness
                score = dot_product / (dist**0.5)
                targets.append((score, i))

        if targets:
            # Sound: Teleport_whoosh.wav
            best_target_idx = max(targets, key=lambda item: item[0])[1]
            self.player_shape_idx = best_target_idx
            self._create_particles(self.shapes[best_target_idx]['pos'], self.COLOR_GHOST, 20)
            return True
        return False

    def _handle_terraform(self, terraform_nearest, terraform_anomaly):
        """Changes the color of the current shape based on player input."""
        current_shape = self.shapes[self.player_shape_idx]
        
        target_color_idx = None
        
        if terraform_anomaly:
            # Sound: Terraform_chime_special.wav
            target_color_idx = -1
        elif terraform_nearest:
            # Sound: Terraform_chime_normal.wav
            current_pos = current_shape['pos']
            closest_dist = float('inf')
            closest_shape_color_idx = current_shape['color_idx']
            
            for i, shape in enumerate(self.shapes):
                if i == self.player_shape_idx:
                    continue
                dist = self._get_distance(current_pos, shape['pos'])
                if dist < closest_dist:
                    closest_dist = dist
                    closest_shape_color_idx = shape['color_idx']
            target_color_idx = closest_shape_color_idx

        if target_color_idx is not None and current_shape['color_idx'] != target_color_idx:
            current_shape['color_idx'] = target_color_idx
            self._create_particles(current_shape['pos'], self.SHAPE_PALETTE[target_color_idx] if target_color_idx != -1 else self.COLOR_ANOMALY, 15, life=15)
            return 1.0 # Reward for successful camouflage action
        return 0.0

    def _check_detection(self):
        """Checks if any scanner detects the player."""
        player_shape = self.shapes[self.player_shape_idx]
        player_pos = player_shape['pos']
        player_color_idx = player_shape['color_idx']
        
        # Player is safe if on an anomaly-colored shape
        if player_color_idx == -1:
            return False

        for scanner in self.scanners:
            dist_to_scanner = self._get_distance(player_pos, scanner['pos'])
            
            if dist_to_scanner < scanner['detection_radius']:
                # If player's color is different from scanner's color, they are detected
                if player_color_idx != scanner['color_idx']:
                    self._create_particles(player_pos, self.COLOR_SCANNER, 50, life=40)
                    return True
        return False

    def _update_scanners(self):
        """Rotates the scanners."""
        speed_increase = (self.steps / 500) * 0.01
        current_speed = self.scanner_base_speed + speed_increase
        for scanner in self.scanners:
            scanner['angle'] += current_speed

    def _update_particles(self):
        """Updates position and lifetime of all particles."""
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
            p['life'] -= 1

    def _create_particles(self, pos, color, count, life=20, speed=2):
        """Generates a burst of particles at a position."""
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = (math.cos(angle) * speed * self.np_random.uniform(0.5, 1.5),
                   math.sin(angle) * speed * self.np_random.uniform(0.5, 1.5))
            self.particles.append({'pos': pos, 'vel': vel, 'life': life, 'color': color})

    def _get_observation(self):
        self._render_to_surface()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_to_surface(self):
        """Main rendering loop to draw the game state to the pygame surface."""
        # Background gradient
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))
        
        # Anomaly
        pulse_radius = self.anomaly['radius'] + 3 * math.sin(self.anomaly_pulse)
        self._draw_glow_circle(self.screen, self.anomaly['pos'], pulse_radius, self.anomaly['color'])

        # Shapes
        for shape in self.shapes:
            color = self.SHAPE_PALETTE[shape['color_idx']] if shape['color_idx'] != -1 else self.COLOR_ANOMALY
            self._draw_shape(self.screen, shape, color)

        # Player Ghost
        player_shape = self.shapes[self.player_shape_idx]
        self._draw_ghost(self.screen, player_shape)
        
        # Scanners
        for scanner in self.scanners:
            self._draw_scanner(self.screen, scanner)
            
        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 20.0))))
            color_with_alpha = p['color'] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 2, color_with_alpha)

        # UI
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()

    def _render_ui(self):
        """Renders the score and other UI elements."""
        score_text = self.font_ui.render(f"SCORE: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        
        dist_text = self.font_ui.render(f"DISTANCE: {self.last_dist_to_anomaly:.0f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(dist_text, (10, 10))

    def _render_game_over(self):
        """Renders the game over message."""
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        text_surface = self.font_game_over.render(self.win_message, True, self.COLOR_UI_TEXT)
        text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surface, text_rect)

    def _draw_shape(self, surface, shape, color):
        """Draws a single geometric shape with anti-aliasing."""
        pos = (int(shape['pos'][0]), int(shape['pos'][1]))
        size = int(shape['size'])
        if shape['type'] == 'circle':
            pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], size, color)
            pygame.gfxdraw.aacircle(surface, pos[0], pos[1], size, color)
        elif shape['type'] == 'square':
            rect = pygame.Rect(pos[0] - size, pos[1] - size, size * 2, size * 2)
            pygame.draw.rect(surface, color, rect)
        elif shape['type'] == 'triangle':
            points = [
                (pos[0], pos[1] - size),
                (pos[0] - size * math.sqrt(3) / 2, pos[1] + size / 2),
                (pos[0] + size * math.sqrt(3) / 2, pos[1] + size / 2)
            ]
            pygame.gfxdraw.filled_polygon(surface, points, color)
            pygame.gfxdraw.aapolygon(surface, points, color)

    def _draw_ghost(self, surface, shape):
        """Draws the player's ghost, mimicking the shape it's on."""
        ghost_color_base = self.COLOR_GHOST
        for i in range(5, 0, -1):
            alpha = 150 - i * 25
            ghost_color = ghost_color_base + (alpha,)
            
            temp_shape = shape.copy()
            temp_shape['size'] = shape['size'] + i
            self._draw_shape(surface, temp_shape, ghost_color)

    def _draw_scanner(self, surface, scanner):
        """Draws a scanner, including its detection radius and rotating line."""
        pos = (int(scanner['pos'][0]), int(scanner['pos'][1]))
        color = self.SHAPE_PALETTE[scanner['color_idx']]
        
        # Draw detection radius
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], scanner['detection_radius'], color + (50,))
        
        # Draw rotating line
        end_x = pos[0] + scanner['length'] * math.cos(scanner['angle'])
        end_y = pos[1] + scanner['length'] * math.sin(scanner['angle'])
        start_x = pos[0] - scanner['length'] * math.cos(scanner['angle'])
        start_y = pos[1] - scanner['length'] * math.sin(scanner['angle'])
        
        pygame.draw.line(surface, self.COLOR_SCANNER, (start_x, start_y), (end_x, end_y), 3)

    def _draw_glow_circle(self, surface, pos, radius, color):
        """Draws a circle with a glowing effect."""
        pos = (int(pos[0]), int(pos[1]))
        for i in range(int(radius), 0, -2):
            alpha = int(150 * (1 - (i / radius))**2)
            glow_color = color + (alpha,)
            pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], i, glow_color)
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], int(radius * 0.7), color)
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], int(radius * 0.7), color)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.shapes[self.player_shape_idx]['pos'],
            "distance_to_anomaly": self.last_dist_to_anomaly,
        }

    @staticmethod
    def _get_distance(pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Obs shape is {test_obs.shape}"
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will create a visible pygame window
    os.environ.pop("SDL_VIDEODRIVER", None)
    env = GameEnv()
    obs, info = env.reset()
    
    print("Controls:")
    print("  Arrows: Teleport")
    print("  Space: Terraform to nearest shape's color")
    print("  Shift: Terraform to anomaly's color (blue)")
    print("  Q: Quit")

    action = [0, 0, 0] # [movement, space, shift]
    
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    while running:
        # Human input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
        
        keys = pygame.key.get_pressed()
        action = [0, 0, 0]
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        render_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(render_surface, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000) # Pause before resetting
            obs, info = env.reset()

        env.clock.tick(GameEnv.TARGET_FPS)
        
    env.close()