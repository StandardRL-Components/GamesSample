import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:07:00.906709
# Source Brief: brief_00222.md
# Brief Index: 222
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where a shape-shifting cube navigates a 3D isometric maze
    to collect crystals against a time limit. The agent must choose the correct form
    (Square, Tetrahedron, Octahedron, Dodecahedron) to optimize its movement and
    jumping capabilities for the terrain.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a 3D isometric maze as a shape-shifting cube. "
        "Collect all the crystals before time runs out, using different forms to optimize movement and jumping."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to change shape and shift to jump."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = FPS * 60  # 60-second time limit

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_WALL = (100, 110, 120)
    COLOR_CRYSTAL = (255, 255, 255)
    COLOR_CRYSTAL_GLOW = (200, 220, 255, 60)
    COLOR_SHADOW = (0, 0, 0, 100)
    COLOR_UI_TEXT = (230, 230, 230)
    COLOR_FORM_SQUARE = (255, 80, 80)
    COLOR_FORM_TETRA = (80, 255, 80)
    COLOR_FORM_OCTA = (80, 80, 255)
    COLOR_FORM_DODECA = (255, 255, 80)

    # Maze and World
    MAZE_SIZE = 12
    TILE_WIDTH = 40
    TILE_HEIGHT = 20
    Z_SCALE = 25

    # Player Physics
    GRAVITY = -0.5
    PLAYER_RADIUS = 0.3  # in grid units

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
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)

        # --- Game State & Entities ---
        self.player_pos = np.zeros(3, dtype=float)  # [x, y, z] in world coords
        self.player_vel = np.zeros(3, dtype=float)
        self.player_form = 0  # 0: Square, 1: Tetra, 2: Octa, 3: Dodeca
        self.on_ground = True
        self.prev_space_held = False
        self.prev_shift_held = False

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.crystals_collected = 0
        self.crystals = []
        self.dist_to_nearest_crystal = 0.0

        # Define form properties: [color, speed_multiplier, jump_velocity]
        self.forms = [
            (self.COLOR_FORM_SQUARE, 0.05, 7.0),   # Square: Slow, High Jump
            (self.COLOR_FORM_TETRA, 0.08, 5.0),    # Tetrahedron: Medium, Medium Jump
            (self.COLOR_FORM_OCTA, 0.12, 3.0),     # Octahedron: Fast, Low Jump
            (self.COLOR_FORM_DODECA, 0.03, 9.0),   # Dodecahedron: Very Slow, Very High Jump
        ]

        # Define a fixed maze layout
        self.maze_layout = self._generate_maze()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset player state
        self.player_pos = np.array([self.MAZE_SIZE / 2.0, self.MAZE_SIZE / 2.0, 0.0])
        self.player_vel = np.zeros(3, dtype=float)
        self.player_form = 0
        self.on_ground = True
        self.prev_space_held = False
        self.prev_shift_held = False

        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.crystals_collected = 0
        self._place_crystals()
        self.dist_to_nearest_crystal = self._find_nearest_crystal_dist()

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Unpack and Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Shape change on button press (rising edge)
        if space_held and not self.prev_space_held:
            self.player_form = (self.player_form + 1) % len(self.forms)
            # SFX: play_shape_change_sound()

        # Jump on button press (rising edge)
        if shift_held and not self.prev_shift_held and self.on_ground:
            _, _, jump_velocity = self.forms[self.player_form]
            self.player_vel[2] = jump_velocity
            self.on_ground = False
            # SFX: play_jump_sound()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- 2. Update Game Logic & Physics ---
        self.steps += 1
        
        # Apply movement based on current form
        _, speed, _ = self.forms[self.player_form]
        move_vec = np.zeros(2, dtype=float)
        if movement == 1: move_vec = np.array([-1, -1]) # Up
        elif movement == 2: move_vec = np.array([1, 1])  # Down
        elif movement == 3: move_vec = np.array([-1, 1])  # Left
        elif movement == 4: move_vec = np.array([1, -1])  # Right
        
        if np.linalg.norm(move_vec) > 0:
            move_vec = move_vec / np.linalg.norm(move_vec) * speed
        
        self.player_vel[0] = move_vec[0]
        self.player_vel[1] = move_vec[1]
        
        # Apply gravity
        if not self.on_ground:
            self.player_vel[2] += self.GRAVITY

        # Update position and handle collisions
        new_pos = self.player_pos + self.player_vel
        
        # Wall collision
        if self.maze_layout[int(new_pos[1])][int(self.player_pos[0])] == 1:
            new_pos[1] = self.player_pos[1] # Block Y
        if self.maze_layout[int(self.player_pos[1])][int(new_pos[0])] == 1:
            new_pos[0] = self.player_pos[0] # Block X

        # Boundary collision
        new_pos[0] = np.clip(new_pos[0], self.PLAYER_RADIUS, self.MAZE_SIZE - self.PLAYER_RADIUS)
        new_pos[1] = np.clip(new_pos[1], self.PLAYER_RADIUS, self.MAZE_SIZE - self.PLAYER_RADIUS)
        self.player_pos = new_pos
        
        # Ground collision
        if self.player_pos[2] < 0:
            self.player_pos[2] = 0
            self.player_vel[2] = 0
            self.on_ground = True

        # --- 3. Check for Events (Crystal Collection) ---
        reward = 0
        collected_this_step = False
        for i, crystal_pos in enumerate(self.crystals):
            dist = np.linalg.norm(self.player_pos - crystal_pos)
            if dist < self.PLAYER_RADIUS + 0.5: # Collection radius
                self.crystals.pop(i)
                self.crystals_collected += 1
                self.score += 1
                reward += 1.0
                collected_this_step = True
                # SFX: play_crystal_collect_sound()
                break
        
        # --- 4. Calculate Reward ---
        new_dist = self._find_nearest_crystal_dist()
        # Reward for getting closer to a crystal
        if not collected_this_step and new_dist is not None:
            reward += (self.dist_to_nearest_crystal - new_dist) * 0.1
        self.dist_to_nearest_crystal = new_dist if new_dist is not None else 0

        # --- 5. Check Termination ---
        terminated = False
        if self.crystals_collected == 10:
            reward += 100
            terminated = True
            self.game_over = True
            # SFX: play_win_sound()
        elif self.steps >= self.MAX_STEPS:
            reward -= 100
            terminated = True
            self.game_over = True
            # SFX: play_lose_sound()

        self.score += reward # Add reward to score for display

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _project(self, x, y, z):
        """Projects 3D world coordinates to 2D screen coordinates."""
        iso_x = (x - y) * self.TILE_WIDTH
        iso_y = (x + y) * self.TILE_HEIGHT
        screen_x = self.SCREEN_WIDTH // 2 + iso_x
        screen_y = self.SCREEN_HEIGHT // 2 + iso_y - z * self.Z_SCALE
        return int(screen_x), int(screen_y)

    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Render game elements
        self._render_game()
        
        # Render UI
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "crystals_collected": self.crystals_collected,
            "player_form": self.player_form,
        }

    def _render_game(self):
        # Create a list of all renderable objects with their depth
        render_list = []

        # Add maze walls
        for r, row in enumerate(self.maze_layout):
            for c, tile in enumerate(row):
                if tile == 1:
                    depth = c + r
                    render_list.append(('wall', (c, r, 0), depth))

        # Add crystals
        for pos in self.crystals:
            depth = pos[0] + pos[1]
            render_list.append(('crystal', pos, depth))

        # Add player
        player_depth = self.player_pos[0] + self.player_pos[1]
        render_list.append(('player', self.player_pos, player_depth))

        # Sort objects by depth for correct occlusion
        render_list.sort(key=lambda item: item[2])
        
        # Draw player shadow first, on the ground plane
        shadow_pos = self.player_pos.copy()
        shadow_pos[2] = 0
        sx, sy = self._project(*shadow_pos)
        shadow_radius = int(15 * (1 - self.player_pos[2] / 20)) # Shadow shrinks with height
        if shadow_radius > 0:
            self._draw_ellipse_alpha(self.screen, self.COLOR_SHADOW, (sx - shadow_radius, sy - shadow_radius//2, shadow_radius*2, shadow_radius))

        # Draw sorted objects
        for obj_type, pos, _ in render_list:
            if obj_type == 'wall':
                self._draw_wall_cube(pos[0], pos[1])
            elif obj_type == 'crystal':
                self._draw_crystal(pos)
            elif obj_type == 'player':
                self._draw_player(pos)

    def _draw_wall_cube(self, x, y):
        # Top face
        p1 = self._project(x, y, 1)
        p2 = self._project(x + 1, y, 1)
        p3 = self._project(x + 1, y + 1, 1)
        p4 = self._project(x, y + 1, 1)
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3, p4], self.COLOR_WALL)
        pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3, p4], self.COLOR_WALL)

    def _draw_crystal(self, pos):
        sx, sy = self._project(*pos)
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, sx, sy, 15, self.COLOR_CRYSTAL_GLOW)
        # Crystal shape (diamond)
        points = [
            (sx, sy - 10), (sx + 6, sy), (sx, sy + 10), (sx - 6, sy)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_CRYSTAL)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_CRYSTAL)

    def _draw_player(self, pos):
        sx, sy = self._project(*pos)
        color, _, _ = self.forms[self.player_form]
        
        # Draw a different 2D shape for each form
        if self.player_form == 0: # Square (isometric cube)
            points = [(sx, sy-10), (sx+10, sy-5), (sx+10, sy+5), (sx, sy+10), (sx-10, sy+5), (sx-10, sy-5)]
        elif self.player_form == 1: # Tetrahedron (triangle)
            points = [(sx, sy-10), (sx+10, sy+8), (sx-10, sy+8)]
        elif self.player_form == 2: # Octahedron (hexagon)
            points = [(sx, sy-12), (sx+10, sy-6), (sx+10, sy+6), (sx, sy+12), (sx-10, sy+6), (sx-10, sy-6)]
        else: # Dodecahedron (circle as approximation)
            pygame.gfxdraw.filled_circle(self.screen, sx, sy, 10, color)
            pygame.gfxdraw.aacircle(self.screen, sx, sy, 10, color)
            return

        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _render_ui(self):
        # Crystals collected
        crystal_text = self.font_small.render(f"Crystals: {self.crystals_collected}/10", True, self.COLOR_UI_TEXT)
        self.screen.blit(crystal_text, (10, 10))

        # Time remaining
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = self.font_small.render(f"Time: {max(0, time_left):.1f}s", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        # Score
        score_text = self.font_large.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH // 2 - score_text.get_width() // 2, self.SCREEN_HEIGHT - 40))

    def _generate_maze(self):
        size = self.MAZE_SIZE
        layout = np.zeros((size, size), dtype=int)
        # Borders
        layout[0, :] = 1
        layout[-1, :] = 1
        layout[:, 0] = 1
        layout[:, -1] = 1
        # Some internal walls
        layout[3, 3:8] = 1
        layout[6, 1:5] = 1
        layout[6, 7:-1] = 1
        layout[9, 4:9] = 1
        return layout

    def _place_crystals(self):
        self.crystals = []
        num_crystals = 10
        while len(self.crystals) < num_crystals:
            x = self.np_random.uniform(1.5, self.MAZE_SIZE - 1.5)
            y = self.np_random.uniform(1.5, self.MAZE_SIZE - 1.5)
            z = self.np_random.uniform(0, 5) # Place some crystals at height
            if self.maze_layout[int(y)][int(x)] == 0:
                self.crystals.append(np.array([x, y, z]))

    def _find_nearest_crystal_dist(self):
        if not self.crystals:
            return None
        distances = [np.linalg.norm(self.player_pos - c) for c in self.crystals]
        return min(distances)

    def _draw_ellipse_alpha(self, surface, color, rect):
        shape_surf = pygame.Surface(pygame.Rect(rect).size, pygame.SRCALPHA)
        pygame.draw.ellipse(shape_surf, color, (0, 0, *pygame.Rect(rect).size))
        surface.blit(shape_surf, rect)

if __name__ == '__main__':
    # --- Manual Play Example ---
    # Un-comment the next line to run with display
    # os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # This check is not needed if running headlessly but good for local testing
    if os.environ.get("SDL_VIDEODRIVER") != "dummy":
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Shape Shifter Maze")
    else:
        screen = None # No display

    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print("Arrows: Move")
    print("Space: Change Shape")
    print("Shift: Jump")
    print("R: Reset")
    print("Q: Quit")

    # Game loop only makes sense with a display
    if screen:
        while not terminated:
            # --- Action Mapping for Manual Play ---
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]

            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                    if event.key == pygame.K_q:
                        terminated = True

            # --- Environment Step ---
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            
            if term:
                print(f"Episode Finished! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
                # Wait for reset or quit
                continue

            # --- Rendering ---
            # The observation is already a rendered frame
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(env.FPS)

    env.close()
    if screen:
        pygame.quit()