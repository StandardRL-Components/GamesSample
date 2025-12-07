import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import os
import pygame


# Set headless mode for pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player manipulates a falling chain of 5
    interconnected blocks. The goal is to make the chain horizontally aligned
    by rotating it and changing the density of individual blocks to alter their
    fall speed. The chain must avoid vertical laser beams.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    game_description = (
        "Manipulate a falling chain of 5 blocks to be horizontally aligned. "
        "Rotate the chain and change block densities to avoid lasers and achieve the goal."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to rotate the chain. "
        "Press space to cycle through blocks and change their density."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 600 # 10 seconds at 60 FPS

    # Colors
    COLOR_BG = (15, 18, 23)
    COLOR_GRID = (30, 35, 45)
    COLOR_LASER = (255, 20, 50)
    COLOR_LASER_GLOW = (180, 20, 50)
    DENSITY_COLORS = [(220, 220, 230), (120, 120, 130), (40, 40, 45)] # Light, Medium, Dark
    COLOR_LINK = (100, 105, 115)
    COLOR_UI_TEXT = (230, 230, 230)
    COLOR_UI_TIMER_WARN = (255, 180, 0)
    COLOR_UI_TIMER_CRIT = (255, 80, 80)
    COLOR_SELECTOR = (255, 255, 255)

    # Physics
    GRAVITY = 0.03
    DENSITY_MODS = [0.8, 1.0, 1.2] # Low density falls faster
    ROTATION_SPEED = 0.08
    BLOCK_SIZE = 20
    REST_LENGTH = BLOCK_SIZE * 2.5
    SPRING_CONSTANT = 0.01
    DAMPING = 0.92

    # 3D Projection
    FOCAL_LENGTH = 350

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.timer = 0.0
        self.blocks = []
        self.laser_x_positions = [self.WIDTH // 3, 2 * self.WIDTH // 3]
        self.prev_space_held = False
        self.density_cycle_idx = 0
        self.aligned_pairs = set()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.timer = self.MAX_STEPS / (self.metadata["render_fps"])
        self.prev_space_held = False
        self.density_cycle_idx = 0
        self.aligned_pairs = set()

        # Initialize the chain of blocks
        self.blocks = []
        # Center the chain horizontally with some randomness.
        # It is initialized vertically to avoid immediate collision with lasers.
        start_x = self.np_random.uniform(-self.WIDTH / 8, self.WIDTH / 8)
        start_y = -self.HEIGHT / 2.5 # Start position from top

        for i in range(5):
            # Create a mostly vertical chain to fit between the lasers at the start.
            pos = np.array([start_x, start_y + (i - 2) * self.BLOCK_SIZE * 2, 0.0], dtype=float)
            self.blocks.append({
                'pos': pos,
                'vel': np.zeros(3, dtype=float),
                'density_idx': 1, # Medium density
            })
        
        # Apply a small random initial rotation for variety
        initial_rot_x = self.np_random.uniform(-0.5, 0.5)
        initial_rot_y = self.np_random.uniform(-0.5, 0.5)
        self._apply_rotation(initial_rot_x, initial_rot_y)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- ACTION HANDLING ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Rotation
        rot_x, rot_y = 0, 0
        if movement == 1: rot_x = self.ROTATION_SPEED   # Up
        elif movement == 2: rot_x = -self.ROTATION_SPEED  # Down
        elif movement == 3: rot_y = self.ROTATION_SPEED   # Left
        elif movement == 4: rot_y = -self.ROTATION_SPEED  # Right
        if rot_x != 0 or rot_y != 0:
            self._apply_rotation(rot_x, rot_y)

        # Density change (on key press, not hold)
        if space_held and not self.prev_space_held:
            block_to_change = self.blocks[self.density_cycle_idx]
            block_to_change['density_idx'] = (block_to_change['density_idx'] + 1) % 3
            self.density_cycle_idx = (self.density_cycle_idx + 1) % 5
        self.prev_space_held = space_held

        # --- GAME LOGIC ---
        self._apply_physics()
        self.steps += 1
        self.timer -= 1 / self.metadata["render_fps"]

        # --- STATE CHECKS & REWARD ---
        terminated, reason = self._check_termination()
        reward = self._calculate_reward(terminated, reason)
        self.score += reward
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False, # truncated
            self._get_info()
        )

    def _apply_rotation(self, rot_x, rot_y):
        # Find center of mass
        com = np.mean([b['pos'] for b in self.blocks], axis=0)
        
        # Rotation matrices
        rx_matrix = np.array([
            [1, 0, 0],
            [0, math.cos(rot_x), -math.sin(rot_x)],
            [0, math.sin(rot_x), math.cos(rot_x)]
        ])
        ry_matrix = np.array([
            [math.cos(rot_y), 0, math.sin(rot_y)],
            [0, 1, 0],
            [-math.sin(rot_y), 0, math.cos(rot_y)]
        ])
        rot_matrix = rx_matrix @ ry_matrix

        for block in self.blocks:
            # Translate to origin, rotate, translate back
            block['pos'] = (rot_matrix @ (block['pos'] - com)) + com

    def _apply_physics(self):
        # 1. Apply gravity
        for block in self.blocks:
            density_mod = self.DENSITY_MODS[block['density_idx']]
            block['vel'][1] += self.GRAVITY * density_mod

        # 2. Apply spring forces between adjacent blocks (multiple iterations for stability)
        for _ in range(5):
            for i in range(len(self.blocks) - 1):
                p1, p2 = self.blocks[i], self.blocks[i+1]
                
                dist_vec = p2['pos'] - p1['pos']
                dist = np.linalg.norm(dist_vec)
                
                if dist == 0: continue # Avoid division by zero
                
                error = dist - self.REST_LENGTH
                force_mag = error * self.SPRING_CONSTANT
                force_vec = (dist_vec / dist) * force_mag

                p1['vel'] += force_vec
                p2['vel'] -= force_vec

        # 3. Apply velocity and damping
        for block in self.blocks:
            block['vel'] *= self.DAMPING
            block['pos'] += block['vel']

    def _project_point(self, point3d):
        # Simple perspective projection
        z = point3d[2] + self.FOCAL_LENGTH
        if z <= 0: return None, None, None # Behind camera
        
        scale = self.FOCAL_LENGTH / z
        x2d = int(self.WIDTH / 2 + point3d[0] * scale)
        y2d = int(self.HEIGHT / 2 + point3d[1] * scale)
        
        return x2d, y2d, scale

    def _check_termination(self):
        # Laser collision
        for block in self.blocks:
            x2d, _, scale = self._project_point(block['pos'])
            if x2d is None: continue
            
            block_radius = self.BLOCK_SIZE * scale
            for laser_x in self.laser_x_positions:
                if abs(x2d - laser_x) < block_radius:
                    return True, "laser"

        # Win condition (horizontal alignment)
        ALIGN_THRESHOLD = 5.0
        y_coords = [b['pos'][1] for b in self.blocks]
        z_coords = [b['pos'][2] for b in self.blocks]
        if (max(y_coords) - min(y_coords)) < ALIGN_THRESHOLD and \
           (max(z_coords) - min(z_coords)) < ALIGN_THRESHOLD:
            return True, "win"

        # Timeout
        if self.timer <= 0 or self.steps >= self.MAX_STEPS:
            return True, "timeout"
            
        return False, "running"

    def _calculate_reward(self, terminated, reason):
        if terminated:
            if reason == "win": return 100.0
            if reason == "laser": return -100.0
            if reason == "timeout": return -10.0
        
        # Continuous rewards
        reward = 0.1  # Survival bonus

        # Alignment bonus
        ALIGN_THRESHOLD_REWARD = 15.0
        current_aligned_pairs = set()
        for i in range(len(self.blocks) - 1):
            p1, p2 = self.blocks[i]['pos'], self.blocks[i+1]['pos']
            y_diff = abs(p1[1] - p2[1])
            z_diff = abs(p1[2] - p2[2])
            if y_diff < ALIGN_THRESHOLD_REWARD and z_diff < ALIGN_THRESHOLD_REWARD:
                current_aligned_pairs.add(i)
        
        newly_aligned = current_aligned_pairs - self.aligned_pairs
        reward += len(newly_aligned) * 1.0
        self.aligned_pairs = current_aligned_pairs

        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "timer": self.timer}

    def _render_game(self):
        # Draw background grid
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

        # Draw lasers
        for x in self.laser_x_positions:
            # Glow
            pygame.draw.line(self.screen, self.COLOR_LASER_GLOW, (x, 0), (x, self.HEIGHT), 7)
            # Core beam
            pygame.draw.line(self.screen, self.COLOR_LASER, (x, 0), (x, self.HEIGHT), 3)

        # Sort blocks by Z for proper rendering order
        sorted_blocks = sorted(enumerate(self.blocks), key=lambda item: item[1]['pos'][2], reverse=True)
        
        # Project all block centers first to draw links
        projected_centers = {}
        for idx, block in sorted_blocks:
            x, y, scale = self._project_point(block['pos'])
            if x is not None:
                projected_centers[idx] = (x, y, scale)

        # Draw links
        for i in range(4):
            if i in projected_centers and i + 1 in projected_centers:
                p1 = projected_centers[i][:2]
                p2 = projected_centers[i+1][:2]
                pygame.draw.line(self.screen, self.COLOR_LINK, p1, p2, 2)

        # Draw blocks
        for idx, block in sorted_blocks:
            if idx not in projected_centers: continue
            x, y, scale = projected_centers[idx]
            
            size = int(self.BLOCK_SIZE * scale)
            if size < 1: continue

            color = self.DENSITY_COLORS[block['density_idx']]
            
            # Draw block as a filled circle for a softer look
            pygame.gfxdraw.filled_circle(self.screen, x, y, size, color)
            pygame.gfxdraw.aacircle(self.screen, x, y, size, color)

            # Draw selector highlight
            if idx == self.density_cycle_idx and not self.game_over:
                pulse = abs(math.sin(self.steps * 0.1))
                radius = int(size * (1.4 + pulse * 0.2))
                alpha = int(100 + pulse * 100)
                s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*self.COLOR_SELECTOR, alpha), (radius, radius), radius, width=max(1, int(3*scale)))
                self.screen.blit(s, (x - radius, y - radius))

    def _render_ui(self):
        # Timer
        timer_str = f"{self.timer:.2f}"
        timer_color = self.COLOR_UI_TEXT
        if self.timer < 5: timer_color = self.COLOR_UI_TIMER_WARN
        if self.timer < 2: timer_color = self.COLOR_UI_TIMER_CRIT
        timer_surf = self.font_large.render(timer_str, True, timer_color)
        self.screen.blit(timer_surf, (self.WIDTH // 2 - timer_surf.get_width() // 2, 10))

        # Score
        score_str = f"Score: {self.score:.1f}"
        score_surf = self.font_small.render(score_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))

    def close(self):
        pygame.quit()

# Example usage for interactive play
if __name__ == '__main__':
    # The "human" render mode is simulated by displaying the rgb_array output.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    # Unset the dummy video driver to allow a display window.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Chain Fall")
    clock = pygame.time.Clock()
    running = True
    
    action = np.array([0, 0, 0]) # [movement, space, shift]

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action[1] = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    action[1] = 0

        # --- Continuous Key Presses ---
        keys = pygame.key.get_pressed()
        action[0] = 0 # No movement
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            action[0] = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action[0] = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action[0] = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action[0] = 4
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render ---
        # The observation is already a rendered frame
        # We just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.1f}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            action = np.array([0, 0, 0])

        clock.tick(env.metadata["render_fps"])
        
    env.close()