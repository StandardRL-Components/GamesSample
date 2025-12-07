import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:28:13.229304
# Source Brief: brief_02191.md
# Brief Index: 2191
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment simulating a bio-organic puzzle game.
    The player navigates a cytoskeleton, building protein filaments (microtubules and actin)
    to traverse platforms, destroy obstacles, and reach the cell membrane at the top.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a cytoskeleton, building protein filaments to traverse platforms, "
        "destroy obstacles, and reach the cell membrane at the top."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move or aim. Press space to build a microtubule (green) "
        "or shift to build an actin filament (red)."
    )
    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_BG = (15, 10, 25)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_PLAYER_GLOW = (255, 255, 0, 40)
    COLOR_MEMBRANE = (180, 255, 255)
    COLOR_MEMBRANE_GLOW = (180, 255, 255, 30)
    COLOR_PLATFORM = (0, 150, 255)
    COLOR_PLATFORM_GLOW = (0, 150, 255, 40)
    COLOR_OBSTACLE = (255, 100, 0)
    COLOR_MICROTUBULE = (100, 255, 100) # Green
    COLOR_ACTIN = (255, 80, 80) # Red
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_BAR = (50, 50, 80)
    COLOR_UI_BAR_FILL = (100, 220, 100)
    
    # Screen and Grid Dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 32
    GRID_ROWS = 20
    CELL_SIZE = SCREEN_WIDTH // GRID_COLS # 20px

    # Game Parameters
    MAX_STEPS = 2000
    INITIAL_MICROTUBULE_RESOURCES = 50
    INITIAL_ACTIN_RESOURCES = 50
    MEMBRANE_Y_POS = CELL_SIZE * 2

    # Entity Types (for grid)
    TYPE_EMPTY = 0
    TYPE_PLATFORM = 1
    TYPE_OBSTACLE = 2
    
    # Filament Types
    FILAMENT_MICROTUBULE = 0
    FILAMENT_ACTIN = 1

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Initialize state variables to None, they will be set in reset()
        self.steps = None
        self.score = None
        self.game_over = None
        self.player_pos = None
        self.player_facing_dir = None
        self.microtubule_resources = None
        self.actin_resources = None
        self.grid = None
        self.filaments = None
        self.particles = None
        self.initial_dist_to_membrane = None
        self.last_dist_to_membrane = None
        self.visited_platforms = None
        self.bg_elements = None
        
        # self.reset() # This is called by the wrapper, not needed here
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_facing_dir = (0, -1) # Start facing up
        
        self._generate_level()
        
        self.microtubule_resources = self.INITIAL_MICROTUBULE_RESOURCES
        self.actin_resources = self.INITIAL_ACTIN_RESOURCES
        
        self.filaments = set()
        self.particles = []
        self.visited_platforms = {tuple(self.player_pos)}

        self.initial_dist_to_membrane = self.player_pos[1]
        self.last_dist_to_membrane = self.player_pos[1]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement_action = action[0]
        build_microtubule_action = action[1] == 1
        build_actin_action = action[2] == 1

        reward = 0
        action_taken = False

        # --- Update Facing Direction ---
        if movement_action == 1: self.player_facing_dir = (0, -1) # Up
        elif movement_action == 2: self.player_facing_dir = (0, 1) # Down
        elif movement_action == 3: self.player_facing_dir = (-1, 0) # Left
        elif movement_action == 4: self.player_facing_dir = (1, 0) # Right

        # --- Handle Actions (Build > Move) ---
        target_pos = (self.player_pos[0] + self.player_facing_dir[0], self.player_pos[1] + self.player_facing_dir[1])

        if build_microtubule_action:
            if self.microtubule_resources > 0 and self._is_valid_build_site(target_pos):
                self._add_filament(self.player_pos, target_pos, self.FILAMENT_MICROTUBULE)
                self.microtubule_resources -= 1
                reward += self._check_for_obstacle_destruction(target_pos)
                # # SFX: build_green.wav
                action_taken = True
        
        if not action_taken and build_actin_action:
            if self.actin_resources > 0 and self._is_valid_build_site(target_pos):
                self._add_filament(self.player_pos, target_pos, self.FILAMENT_ACTIN)
                self.actin_resources -= 1
                reward += self._check_for_obstacle_destruction(target_pos)
                # # SFX: build_red.wav
                action_taken = True

        if not action_taken and movement_action != 0:
            move_dir = self.player_facing_dir
            new_pos = (self.player_pos[0] + move_dir[0], self.player_pos[1] + move_dir[1])
            
            if self._is_valid_move(self.player_pos, new_pos):
                self.player_pos = new_pos
                # # SFX: move.wav
                
                # Check for obstacle collision
                if self.grid[new_pos[1], new_pos[0]] == self.TYPE_OBSTACLE:
                    self.game_over = True
                    reward -= 5
                    self._create_particles(self._grid_to_pixel(new_pos), self.COLOR_PLAYER, 50)
                    # # SFX: player_death.wav
                
                # Check for reaching a new platform
                elif self.grid[new_pos[1], new_pos[0]] == self.TYPE_PLATFORM and tuple(new_pos) not in self.visited_platforms:
                    self.visited_platforms.add(tuple(new_pos))
                    reward += 10
                    # # SFX: platform_activate.wav
        
        # --- Update Game State & Rewards ---
        self.steps += 1
        
        # Distance-based reward
        current_dist = self.player_pos[1]
        if current_dist < self.last_dist_to_membrane:
            reward += 0.1
        elif current_dist > self.last_dist_to_membrane:
            reward -= 0.1
        self.last_dist_to_membrane = current_dist

        self.score += reward

        # --- Check Termination Conditions ---
        terminated = self.game_over
        truncated = False
        if self._is_in_membrane(self.player_pos):
            self.score += 100
            reward += 100
            terminated = True
            # # SFX: win_level.wav
        
        if self.steps >= self.MAX_STEPS:
            truncated = True
            terminated = True
            
        # Check for being stuck with no resources
        if self.microtubule_resources == 0 and self.actin_resources == 0 and not self._can_move_anywhere():
             terminated = True
             
        self.game_over = terminated

        self._update_particles()

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _generate_level(self):
        self.grid = np.full((self.GRID_ROWS, self.GRID_COLS), self.TYPE_EMPTY, dtype=int)
        
        # Generate background elements
        self.bg_elements = []
        for _ in range(40):
            self.bg_elements.append({
                "pos": (random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT)),
                "radius": random.randint(50, 150),
                "color": (
                    random.randint(20, 30),
                    random.randint(15, 25),
                    random.randint(35, 45),
                    random.randint(20, 40)
                )
            })

        # Create a guaranteed path of platforms
        path = []
        start_x = self.GRID_COLS // 2
        start_y = self.GRID_ROWS - 2
        self.player_pos = [start_x, start_y]
        path.append(tuple(self.player_pos))
        self.grid[start_y, start_x] = self.TYPE_PLATFORM

        current_pos = [start_x, start_y]
        while current_pos[1] > 3:
            move_y = -1
            move_x = random.choice([-2, -1, 0, 1, 2])
            next_pos = [
                max(1, min(self.GRID_COLS - 2, current_pos[0] + move_x)),
                max(3, current_pos[1] + move_y)
            ]
            if tuple(next_pos) not in path:
                self.grid[next_pos[1], next_pos[0]] = self.TYPE_PLATFORM
                path.append(tuple(next_pos))
                current_pos = next_pos

        # Add some extra platforms
        for _ in range(15):
            x, y = random.randint(1, self.GRID_COLS - 2), random.randint(3, self.GRID_ROWS - 2)
            if self.grid[y, x] == self.TYPE_EMPTY:
                self.grid[y, x] = self.TYPE_PLATFORM

        # Add obstacles, avoiding platforms and a buffer around them
        obstacle_density = 0.05 + (self.steps / self.MAX_STEPS) * 0.15 # Difficulty scaling
        for r in range(3, self.GRID_ROWS - 1):
            for c in range(1, self.GRID_COLS - 1):
                is_near_platform = False
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        if self.grid[r+dr, c+dc] == self.TYPE_PLATFORM:
                            is_near_platform = True
                            break
                    if is_near_platform: break
                
                if not is_near_platform and random.random() < obstacle_density:
                    self.grid[r, c] = self.TYPE_OBSTACLE
    
    def _is_valid_build_site(self, pos):
        x, y = pos
        if not (0 <= x < self.GRID_COLS and 0 <= y < self.GRID_ROWS):
            return False
        return self.grid[y, x] == self.TYPE_EMPTY

    def _is_valid_move(self, from_pos, to_pos):
        x, y = to_pos
        if not (0 <= x < self.GRID_COLS and 0 <= y < self.GRID_ROWS):
            return False
        
        # Can always move to a platform
        if self.grid[y, x] == self.TYPE_PLATFORM:
            return True
            
        # Can move along filaments
        # A filament must exist between from_pos and to_pos
        if (tuple(from_pos), tuple(to_pos)) in self.filaments or \
           (tuple(to_pos), tuple(from_pos)) in self.filaments:
            return True
        
        return False

    def _can_move_anywhere(self):
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            target_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
            if self._is_valid_move(self.player_pos, target_pos):
                return True
        return False

    def _add_filament(self, pos1, pos2, f_type):
        segment = tuple(sorted((tuple(pos1), tuple(pos2))))
        self.filaments.add(segment + (f_type,))

    def _check_for_obstacle_destruction(self, new_filament_pos):
        reward = 0
        for dx, dy in [(x, y) for x in range(-1, 2) for y in range(-1, 2) if not (x==0 and y==0)]:
            check_pos = (new_filament_pos[0] + dx, new_filament_pos[1] + dy)
            if not (0 <= check_pos[0] < self.GRID_COLS and 0 <= check_pos[1] < self.GRID_ROWS):
                continue
            
            if self.grid[check_pos[1], check_pos[0]] == self.TYPE_OBSTACLE:
                micro_count = 0
                actin_count = 0
                
                # Check neighbors of the obstacle
                for ob_dx, ob_dy in [(x, y) for x in range(-1, 2) for y in range(-1, 2) if not (x==0 and y==0)]:
                    neighbor_pos = (check_pos[0] + ob_dx, check_pos[1] + ob_dy)
                    # Check if a filament connects to the neighbor from the obstacle
                    segment_key1 = tuple(sorted((check_pos, neighbor_pos)))
                    for f in self.filaments:
                        if tuple(sorted((f[0], f[1]))) == segment_key1:
                            if f[2] == self.FILAMENT_MICROTUBULE: micro_count += 1
                            elif f[2] == self.FILAMENT_ACTIN: actin_count += 1
                            break # Found the filament for this neighbor pair
                
                if micro_count >= 2 and actin_count >= 1:
                    self.grid[check_pos[1], check_pos[0]] = self.TYPE_EMPTY
                    reward += 5
                    # # SFX: obstacle_destroy.wav
                    self._create_particles(self._grid_to_pixel(check_pos), self.COLOR_OBSTACLE, 100)
        return reward

    def _is_in_membrane(self, pos):
        px_y = self._grid_to_pixel(pos)[1]
        return px_y <= self.MEMBRANE_Y_POS

    def _grid_to_pixel(self, grid_pos):
        x = int(grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2)
        y = int(grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2)
        return x, y
    
    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": random.randint(20, 40),
                "color": color
            })
            
    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.95 # friction
            p["vel"][1] *= 0.95
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background cytosol
        for bg in self.bg_elements:
            pygame.gfxdraw.filled_circle(self.screen, int(bg["pos"][0]), int(bg["pos"][1]), bg["radius"], bg["color"])

        # Cell Membrane
        membrane_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, self.MEMBRANE_Y_POS)
        pygame.draw.rect(self.screen, self.COLOR_MEMBRANE, membrane_rect, border_bottom_left_radius=15, border_bottom_right_radius=15)
        for i in range(20):
            glow_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, self.MEMBRANE_Y_POS + i*2)
            pygame.draw.rect(self.screen, self.COLOR_MEMBRANE_GLOW, glow_rect, width=2, border_bottom_left_radius=15, border_bottom_right_radius=15)

        # Filaments
        for filament_data in self.filaments:
            start_pos, end_pos, f_type = filament_data[0], filament_data[1], filament_data[2]
            p1 = self._grid_to_pixel(start_pos)
            p2 = self._grid_to_pixel(end_pos)
            color = self.COLOR_MICROTUBULE if f_type == self.FILAMENT_MICROTUBULE else self.COLOR_ACTIN
            pygame.draw.line(self.screen, color, p1, p2, 3)

        # Platforms and Obstacles
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                entity_type = self.grid[r, c]
                if entity_type == self.TYPE_EMPTY: continue
                
                pos_px = self._grid_to_pixel((c, r))
                
                if entity_type == self.TYPE_PLATFORM:
                    radius = self.CELL_SIZE // 3
                    pygame.gfxdraw.filled_circle(self.screen, pos_px[0], pos_px[1], radius + 5, self.COLOR_PLATFORM_GLOW)
                    pygame.gfxdraw.filled_circle(self.screen, pos_px[0], pos_px[1], radius, self.COLOR_PLATFORM)
                    pygame.gfxdraw.aacircle(self.screen, pos_px[0], pos_px[1], radius, self.COLOR_PLATFORM)
                
                elif entity_type == self.TYPE_OBSTACLE:
                    size = self.CELL_SIZE // 2.5
                    points = [
                        (pos_px[0] + size * math.cos(math.pi * 2 * i / 5 + math.pi/10), 
                         pos_px[1] + size * math.sin(math.pi * 2 * i / 5 + math.pi/10)) 
                        for i in range(5)
                    ]
                    pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_OBSTACLE)
                    pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_OBSTACLE)
        
        # Particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / 40))
            # Create a temporary surface for alpha blending
            temp_surf = pygame.Surface((p['life']/5, p['life']/5), pygame.SRCALPHA)
            color_with_alpha = (*p['color'], alpha)
            pygame.draw.circle(temp_surf, color_with_alpha, (p['life']/10, p['life']/10), p['life'] / 10)
            self.screen.blit(temp_surf, (p['pos'][0] - p['life']/10, p['pos'][1] - p['life']/10))


        # Player
        if not self.game_over:
            player_px = self._grid_to_pixel(self.player_pos)
            radius = self.CELL_SIZE // 2.5
            pygame.gfxdraw.filled_circle(self.screen, player_px[0], player_px[1], int(radius * 2), self.COLOR_PLAYER_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, player_px[0], player_px[1], int(radius), self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, player_px[0], player_px[1], int(radius), self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Resources
        micro_text = self.font_small.render(f"Microtubules: {self.microtubule_resources}", True, self.COLOR_MICROTUBULE)
        actin_text = self.font_small.render(f"Actin Filaments: {self.actin_resources}", True, self.COLOR_ACTIN)
        self.screen.blit(micro_text, (self.SCREEN_WIDTH - micro_text.get_width() - 10, 10))
        self.screen.blit(actin_text, (self.SCREEN_WIDTH - actin_text.get_width() - 10, 30))

        # Distance to Membrane Bar
        progress = 1.0 - max(0, min(1, (self.player_pos[1] - self.MEMBRANE_Y_POS/self.CELL_SIZE) / (self.initial_dist_to_membrane - self.MEMBRANE_Y_POS/self.CELL_SIZE)))
        bar_width = 200
        bar_height = 15
        bar_x = (self.SCREEN_WIDTH - bar_width) // 2
        bar_y = 15
        
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        fill_width = int(bar_width * progress)
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_UI_BAR_FILL, (bar_x, bar_y, fill_width, bar_height), border_radius=4)
        
        dist_text = self.font_small.render("PROXIMITY TO MEMBRANE", True, self.COLOR_UI_TEXT)
        self.screen.blit(dist_text, (bar_x + (bar_width - dist_text.get_width()) // 2, bar_y + bar_height))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "microtubule_resources": self.microtubule_resources,
            "actin_resources": self.actin_resources,
            "player_pos": self.player_pos,
        }

    def close(self):
        pygame.quit()
        

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        print("Cannot run main block in dummy mode. Exiting.")
        exit()
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Cytoskeleton Navigator")
    clock = pygame.time.Clock()
    
    terminated = False
    
    # --- Action mapping for human play ---
    # ARROWS: Set direction
    # SPACE: Build microtubule (green)
    # SHIFT: Build actin filament (red)
    # NO KEY: Move in last direction
    
    last_move_action = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
    
    print("\n--- Controls ---")
    print(GameEnv.user_guide)
    print("R: Reset environment")
    print("Q: Quit")
    print("----------------\n")
    
    while not terminated:
        # Default action is to move in the last direction
        action = [last_move_action, 0, 0] 
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    terminated = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    print(f"Environment Reset. Score: {info['score']}")

        keys = pygame.key.get_pressed()
        
        move_action = 0
        if keys[pygame.K_UP]: move_action = 1
        elif keys[pygame.K_DOWN]: move_action = 2
        elif keys[pygame.K_LEFT]: move_action = 3
        elif keys[pygame.K_RIGHT]: move_action = 4
        
        if move_action != 0:
            last_move_action = move_action
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        # If a build key is held, the arrow keys are for aiming, not moving
        if space_held or shift_held:
            action = [last_move_action, space_held, shift_held]
        else: # Otherwise, arrow keys mean move
            action = [move_action, 0, 0]

        obs, reward, term, trunc, info = env.step(action)
        terminated = term or trunc
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']:.1f}, Reward: {reward:.1f}, Terminated: {term}, Truncated: {trunc}")

        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(10) # Control game speed for human play

    print("Game Over!")
    env.close()