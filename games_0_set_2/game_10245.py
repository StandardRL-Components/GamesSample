import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:03:48.974203
# Source Brief: brief_00245.md
# Brief Index: 245
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a dreamlike puzzle/stealth game.

    The player controls a cursor on a circular board to place clones of
    "dream fragments". The goal is to place a clone on the central "Dream Artifact"
    without being detected by patrolling "Dream Guardians".

    **Visuals:**
    - Ethereal, glowing geometric shapes on a dark, starry background.
    - Smooth animations and particle effects for a polished feel.

    **Gameplay:**
    - The game is turn-based.
    - Move a placement cursor around a polar grid.
    - Cycle through different types of dream fragments to clone.
    - Place clones on the board to act as obstacles or to capture the artifact.
    - Guardians patrol in circular paths and will detect clones placed too close.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right) - moves the cursor.
    - actions[1]: Space button (0=released, 1=held) - places a clone.
    - actions[2]: Shift button (0=released, 1=held) - cycles fragment type.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "In this dreamlike puzzle/stealth game, place clones of 'dream fragments' on a circular board "
        "to capture the central artifact while avoiding patrolling guardians."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move the cursor. Press Space to place a clone and Shift to cycle fragment types."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CENTER_X, CENTER_Y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
    BOARD_RADIUS = 160
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_BOARD_LINE = (50, 40, 90, 100)
    COLOR_ARTIFACT = (255, 223, 0)
    COLOR_GUARDIAN = (255, 50, 50)
    COLOR_GUARDIAN_VISION = (255, 50, 50, 25)
    COLOR_CURSOR = (200, 200, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    FRAGMENT_COLORS = [(0, 255, 255), (0, 255, 128), (255, 0, 255)] # Blue, Green, Magenta

    # Board Grid
    RADIAL_DIVISIONS = 5
    ANGULAR_DIVISIONS = 12
    TILE_RADIUS = BOARD_RADIUS / RADIAL_DIVISIONS

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""

        # Entities
        self.cursor_pos = [0, 0]  # [angular_idx, radial_idx]
        self.guardians = []
        self.clones = []
        self.particles = []

        # Player State
        self.selected_fragment_idx = 0
        self.clones_remaining = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.last_cursor_dist_to_artifact = 0.0
        self.last_cursor_dist_to_guardian = 0.0

        # Starry background
        self.stars = [
            (random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT), random.randint(1, 2))
            for _ in range(150)
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""

        # Player state
        self.cursor_pos = [0, self.RADIAL_DIVISIONS - 1] # Start at edge
        self.selected_fragment_idx = 0
        self.clones_remaining = 10
        self.prev_space_held = True # Prevent action on first frame
        self.prev_shift_held = True

        # Entities
        self.clones.clear()
        self.particles.clear()
        self.guardians = [
            {'angle': math.pi / 2, 'radius_idx': 2, 'speed': 0.02, 'direction': 1},
        ]
        
        self._update_distance_metrics()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        self.steps += 1
        
        if self.game_over:
            # If the game was already over, just return the current state
            return self._get_observation(), 0.0, self.game_over, False, self._get_info()

        # --- Process Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        # 1. Handle Cursor Movement
        if movement == 1: # Up (towards center)
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: # Down (away from center)
            self.cursor_pos[1] = min(self.RADIAL_DIVISIONS - 1, self.cursor_pos[1] + 1)
        elif movement == 3: # Left (counter-clockwise)
            self.cursor_pos[0] = (self.cursor_pos[0] - 1) % self.ANGULAR_DIVISIONS
        elif movement == 4: # Right (clockwise)
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.ANGULAR_DIVISIONS

        # 2. Handle Fragment Selection (Shift)
        if shift_pressed:
            self.selected_fragment_idx = (self.selected_fragment_idx + 1) % len(self.FRAGMENT_COLORS)

        # 3. Handle Clone Placement (Space)
        if space_pressed:
            reward += self._place_clone()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Update Game World ---
        if not self.game_over:
            self._update_guardians()
            self._update_particles()
        
        # --- Calculate Rewards ---
        reward += self._calculate_continuous_reward()
        self._update_distance_metrics()

        self.score += reward
        
        # --- Check for End Conditions ---
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        if truncated and not terminated:
            self.game_over = True
            self.win_message = "TIME LIMIT REACHED"
            reward -= 50 # Penalty for timeout
            self.score -= 50

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _place_clone(self):
        if self.clones_remaining <= 0:
            return -1.0 # Failed clone attempt

        # Check for invalid placement (on another clone or guardian)
        for clone in self.clones:
            if clone['pos'] == self.cursor_pos:
                return -1.0
        for guardian in self.guardians:
            guardian_pos = self._polar_to_grid(guardian['angle'], guardian['radius_idx'])
            if guardian_pos == self.cursor_pos:
                return -1.0

        # Successful placement
        new_clone = {
            'pos': list(self.cursor_pos),
            'type_idx': self.selected_fragment_idx,
            'creation_time': self.steps,
            'pulse_phase': random.uniform(0, 2 * math.pi)
        }
        self.clones.append(new_clone)
        self.clones_remaining -= 1
        self._create_particles_at_pos(self.cursor_pos, self.FRAGMENT_COLORS[self.selected_fragment_idx])

        # Check for win condition (clone on artifact)
        if self.cursor_pos[1] == 0: # Radius 0 is the center
            self.game_over = True
            self.win_message = "ARTIFACT STOLEN!"
            return 100.0

        # Check for detection by guardians
        for guardian in self.guardians:
            dist = self._grid_distance(self.cursor_pos, self._polar_to_grid(guardian['angle'], guardian['radius_idx']))
            if dist < 2.0: # Detection radius
                self.game_over = True
                self.win_message = "DETECTED!"
                return -100.0

        return 5.0 # Successful clone reward

    def _update_guardians(self):
        # Increase speed over time
        speed_increase = 0.05 * (self.steps // 200)
        
        for g in self.guardians:
            g['angle'] += (g['speed'] + speed_increase) * g['direction']
            # Simple collision with clones: reverse direction
            g_pos_grid = self._polar_to_grid(g['angle'], g['radius_idx'])
            for clone in self.clones:
                if clone['pos'] == g_pos_grid:
                    g['direction'] *= -1
                    break

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _calculate_continuous_reward(self):
        reward = 0.0
        # Reward for moving cursor closer to artifact (center)
        current_dist_artifact = self.cursor_pos[1] # radial index is distance
        if current_dist_artifact < self.last_cursor_dist_to_artifact:
            reward += 0.1
        elif current_dist_artifact > self.last_cursor_dist_to_artifact:
            reward -= 0.1

        # Penalty for moving cursor closer to a guardian
        if self.guardians:
            dists_to_guardians = [self._grid_distance(self.cursor_pos, self._polar_to_grid(g['angle'], g['radius_idx'])) for g in self.guardians]
            current_dist_guardian = min(dists_to_guardians) if dists_to_guardians else float('inf')
            if current_dist_guardian < self.last_cursor_dist_to_guardian:
                reward -= 0.5 # Stronger penalty for getting closer to danger
        
        return reward

    def _update_distance_metrics(self):
        self.last_cursor_dist_to_artifact = self.cursor_pos[1]
        if self.guardians:
            dists = [self._grid_distance(self.cursor_pos, self._polar_to_grid(g['angle'], g['radius_idx'])) for g in self.guardians]
            self.last_cursor_dist_to_guardian = min(dists) if dists else float('inf')
        else:
            self.last_cursor_dist_to_guardian = float('inf')

    def _polar_to_cartesian(self, angle, radius_val):
        x = self.CENTER_X + radius_val * math.cos(angle)
        y = self.CENTER_Y + radius_val * math.sin(angle)
        return int(x), int(y)

    def _grid_to_polar(self, grid_pos):
        angle = (grid_pos[0] / self.ANGULAR_DIVISIONS) * 2 * math.pi
        radius_val = (grid_pos[1] + 0.5) * self.TILE_RADIUS
        return angle, radius_val

    def _polar_to_grid(self, angle, radius_idx):
        angle_idx = int((angle / (2 * math.pi)) * self.ANGULAR_DIVISIONS) % self.ANGULAR_DIVISIONS
        return [angle_idx, radius_idx]

    def _grid_distance(self, grid_pos1, grid_pos2):
        # Use Manhattan distance on the grid as a simple metric
        d_angle = abs(grid_pos1[0] - grid_pos2[0])
        d_angle = min(d_angle, self.ANGULAR_DIVISIONS - d_angle) # Account for wrap-around
        d_radius = abs(grid_pos1[1] - grid_pos2[1])
        return d_angle + d_radius

    def _create_particles_at_pos(self, grid_pos, color):
        angle, radius_val = self._grid_to_polar(grid_pos)
        cx, cy = self._polar_to_cartesian(angle, radius_val)
        for _ in range(20):
            p_angle = random.uniform(0, 2 * math.pi)
            p_speed = random.uniform(0.5, 2.0)
            self.particles.append({
                'pos': [cx, cy],
                'vel': [math.cos(p_angle) * p_speed, math.sin(p_angle) * p_speed],
                'life': random.randint(15, 30),
                'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # 1. Stars
        for x, y, size in self.stars:
            twinkle = random.randint(150, 255)
            pygame.draw.rect(self.screen, (twinkle, twinkle, twinkle), (x, y, size, size))

        # 2. Board Grid
        for i in range(1, self.RADIAL_DIVISIONS + 1):
            pygame.gfxdraw.aacircle(self.screen, self.CENTER_X, self.CENTER_Y, int(i * self.TILE_RADIUS), self.COLOR_BOARD_LINE)
        for i in range(self.ANGULAR_DIVISIONS):
            angle = (i / self.ANGULAR_DIVISIONS) * 2 * math.pi
            x2 = self.CENTER_X + self.BOARD_RADIUS * math.cos(angle)
            y2 = self.CENTER_Y + self.BOARD_RADIUS * math.sin(angle)
            pygame.draw.aaline(self.screen, self.COLOR_BOARD_LINE, (self.CENTER_X, self.CENTER_Y), (x2, y2))
            
        # 3. Artifact
        pulse = (math.sin(self.steps * 0.1) + 1) / 2
        artifact_size = int(8 + pulse * 6)
        pygame.gfxdraw.filled_circle(self.screen, self.CENTER_X, self.CENTER_Y, artifact_size, self.COLOR_ARTIFACT)
        pygame.gfxdraw.aacircle(self.screen, self.CENTER_X, self.CENTER_Y, artifact_size, self.COLOR_ARTIFACT)
        pygame.gfxdraw.aacircle(self.screen, self.CENTER_X, self.CENTER_Y, int(artifact_size * (1.5 + pulse*0.5)), (*self.COLOR_ARTIFACT, 50))

        # 4. Clones
        for clone in self.clones:
            angle, radius_val = self._grid_to_polar(clone['pos'])
            cx, cy = self._polar_to_cartesian(angle, radius_val)
            color = self.FRAGMENT_COLORS[clone['type_idx']]
            pulse = (math.sin(self.steps * 0.1 + clone['pulse_phase']) + 1) / 2
            size = int(self.TILE_RADIUS * 0.3 + pulse * 2)
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, size, color)
            pygame.gfxdraw.aacircle(self.screen, cx, cy, size, color)
            pygame.gfxdraw.aacircle(self.screen, cx, cy, int(size * 1.8), (*color, 60))

        # 5. Guardians
        for g in self.guardians:
            radius_val = (g['radius_idx'] + 0.5) * self.TILE_RADIUS
            gx, gy = self._polar_to_cartesian(g['angle'], radius_val)
            # Vision cone
            vision_angle_span = math.pi / 4
            points = [(gx, gy)]
            for i in range(11):
                a = g['angle'] + g['direction'] * (-vision_angle_span / 2 + (vision_angle_span * i / 10))
                r = self.TILE_RADIUS * 2.5
                points.append((gx + r * math.cos(a), gy + r * math.sin(a)))
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GUARDIAN_VISION)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_GUARDIAN_VISION)
            # Guardian body
            pulse = (math.sin(self.steps * 0.2) + 1) / 2
            size = int(self.TILE_RADIUS * 0.35 + pulse * 3)
            points = [
                (gx + size * math.cos(g['angle'] + i * 2 * math.pi / 3), gy + size * math.sin(g['angle'] + i * 2 * math.pi / 3))
                for i in range(3)
            ]
            pygame.gfxdraw.filled_trigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), self.COLOR_GUARDIAN)
            pygame.gfxdraw.aatrigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), self.COLOR_GUARDIAN)
            
        # 6. Cursor
        if not self.game_over:
            angle, radius_val = self._grid_to_polar(self.cursor_pos)
            cx, cy = self._polar_to_cartesian(angle, radius_val)
            pulse = (math.sin(self.steps * 0.2) + 1) / 2
            size = int(self.TILE_RADIUS * 0.45 + pulse * 2)
            pygame.gfxdraw.aacircle(self.screen, cx, cy, size, self.COLOR_CURSOR)
            pygame.gfxdraw.aacircle(self.screen, cx, cy, size-1, self.COLOR_CURSOR)

        # 7. Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            color = (*p['color'], alpha)
            pygame.draw.circle(self.screen, color, (int(p['pos'][0]), int(p['pos'][1])), int(p['life'] * 0.1 + 1))

    def _render_ui(self):
        # Score and step text
        score_text = self.font_small.render(f"SCORE: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (10, 30))

        # Clone info
        clones_text = self.font_small.render(f"CLONES LEFT: {self.clones_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(clones_text, (self.SCREEN_WIDTH - clones_text.get_width() - 10, 10))
        
        # Selected Fragment Indicator
        frag_text = self.font_small.render("SELECTED:", True, self.COLOR_UI_TEXT)
        self.screen.blit(frag_text, (self.SCREEN_WIDTH - frag_text.get_width() - 50, 30))
        color = self.FRAGMENT_COLORS[self.selected_fragment_idx]
        pygame.gfxdraw.filled_circle(self.screen, self.SCREEN_WIDTH - 30, 38, 10, color)
        pygame.gfxdraw.aacircle(self.screen, self.SCREEN_WIDTH - 30, 38, 10, color)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.win_message, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.CENTER_X, self.CENTER_Y))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "clones_remaining": self.clones_remaining,
            "cursor_pos": self.cursor_pos,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Manual Play Loop ---
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Override screen for direct display
    env.screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Dream Thief")

    action = [0, 0, 0] # No-op
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Q: Quit")

    clock = pygame.time.Clock()
    
    while not done:
        # Action mapping from keyboard
        action = [0, 0, 0] # Reset to no-op
        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4

        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                done = True
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render to the display
        # The observation is already rendered to env.screen, so just flip the display
        pygame.display.flip()
        clock.tick(30) # Limit to 30 FPS for smooth viewing

    env.close()
    print(f"Game Over. Final Score: {info['score']:.2f}")