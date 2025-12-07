
# Generated: 2025-08-28T04:42:56.987588
# Source Brief: brief_02389.md
# Brief Index: 2389

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Hold Space near a clue to collect it."
    )

    game_description = (
        "Explore a procedurally generated, foggy town to find 5 clues before time runs out. "
        "Beware of hazardous zones. Complete 3 stages to solve the mystery."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 30, 30
        self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF = 24, 12
        self.WORLD_OFFSET_X = self.WIDTH // 2
        self.WORLD_OFFSET_Y = 60

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 18)
        self.font_small = pygame.font.SysFont("Consolas", 14)

        # Colors
        self.COLOR_BG = (15, 18, 24)
        self.COLOR_TILE = (40, 45, 55)
        self.COLOR_TILE_EDGE = (30, 35, 45)
        self.COLOR_BUILDING_A = (60, 65, 75)
        self.COLOR_BUILDING_B = (50, 55, 65)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_CLUE = (0, 255, 128)
        self.COLOR_PENALTY = (255, 50, 50)
        self.COLOR_FOG = (80, 85, 95)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_UI_SUCCESS = (0, 255, 128)
        self.COLOR_UI_FAIL = (255, 80, 80)
        
        # Game state variables are initialized in reset()
        self.player_pos = None
        self.player_screen_pos = None
        self.player_target_pos = None
        self.player_is_moving = False
        self.world_grid = None
        self.clues = None
        self.penalty_zones = None
        self.fog_zones = None
        self.particles = None
        self.streetlights = None
        self.explored_tiles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.current_stage = None
        self.time_remaining = None
        self.clues_collected_this_stage = None
        self.last_action_was_interact = False
        self.message_text = None
        self.message_timer = None
        
        self.reset()
        self.validate_implementation()

    def _grid_to_screen(self, x, y):
        screen_x = (x - y) * self.TILE_WIDTH_HALF + self.WORLD_OFFSET_X
        screen_y = (x + y) * self.TILE_HEIGHT_HALF + self.WORLD_OFFSET_Y
        return int(screen_x), int(screen_y)

    def _generate_stage(self):
        # Generate a valid, traversable map
        while True:
            self.world_grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
            num_buildings = self.GRID_WIDTH * self.GRID_HEIGHT // 6
            for _ in range(num_buildings):
                w, h = self.np_random.integers(2, 5, size=2)
                x, y = self.np_random.integers(0, self.GRID_WIDTH - w), self.np_random.integers(0, self.GRID_HEIGHT - h)
                self.world_grid[x:x+w, y:y+h] = 1

            # Player start position
            start_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT - 2)
            if self.world_grid[start_pos] == 1: continue

            # Flood fill to find reachable tiles
            q = deque([start_pos])
            reachable = {start_pos}
            while q:
                cx, cy = q.popleft()
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.world_grid[nx, ny] == 0 and (nx, ny) not in reachable:
                        reachable.add((nx, ny))
                        q.append((nx, ny))
            
            if len(reachable) < 50: continue # Ensure enough open space

            # Place clues
            self.clues = []
            possible_clue_locs = list(reachable - {start_pos})
            if len(possible_clue_locs) < 5: continue
            
            clue_indices = self.np_random.choice(len(possible_clue_locs), 5, replace=False)
            for i in clue_indices:
                self.clues.append(list(possible_clue_locs[i]))
            
            self.player_pos = list(start_pos)
            self.player_target_pos = list(start_pos)
            self.player_screen_pos = list(self._grid_to_screen(*self.player_pos))
            break
        
        # Difficulty scaling
        fog_density = 0.1 * (self.current_stage - 1)
        penalty_size = 5 + int(5 * (self.current_stage - 1) * 0.05)
        
        # Place zones
        self.penalty_zones = []
        for _ in range(penalty_size):
            if reachable:
                self.penalty_zones.append(random.choice(list(reachable)))

        self.fog_zones = []
        for tile in reachable:
            if self.np_random.random() < fog_density:
                self.fog_zones.append(tile)
        
        # Place streetlights
        self.streetlights = []
        for _ in range(15):
            if reachable:
                self.streetlights.append(random.choice(list(reachable)))

    def _setup_stage(self):
        self.time_remaining = 60 * 30  # 60 seconds at 30fps
        self.clues_collected_this_stage = 0
        self._generate_stage()
        self.explored_tiles = {tuple(self.player_pos)}
        self.particles = []
        self.player_is_moving = False
        self.last_action_was_interact = False
        self._set_message(f"Stage {self.current_stage}", self.COLOR_UI_TEXT, 90)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_stage = 1
        self._setup_stage()
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.clock.tick(30)
        self.steps += 1
        self.time_remaining -= 1
        reward = 0

        # Unpack action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Player Movement ---
        if not self.player_is_moving:
            move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # up, down, left, right
            if movement in move_map:
                dx, dy = move_map[movement]
                next_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
                if (0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT and self.world_grid[next_pos] == 0):
                    self.player_target_pos = list(next_pos)
                    self.player_is_moving = True
                    # Sound: Footstep
        
        # --- Smooth Movement Interpolation ---
        if self.player_is_moving:
            target_screen_pos = self._grid_to_screen(*self.player_target_pos)
            dx = target_screen_pos[0] - self.player_screen_pos[0]
            dy = target_screen_pos[1] - self.player_screen_pos[1]
            dist = math.hypot(dx, dy)
            
            move_speed = 4.0
            if dist < move_speed:
                self.player_screen_pos = list(target_screen_pos)
                self.player_pos = list(self.player_target_pos)
                self.player_is_moving = False

                # Exploration reward
                if tuple(self.player_pos) not in self.explored_tiles:
                    reward += 0.1
                    self.explored_tiles.add(tuple(self.player_pos))
                else:
                    reward -= 0.01 # Small penalty for revisiting, not as harsh as brief
                
                # Penalty zone check
                if tuple(self.player_pos) in self.penalty_zones:
                    reward -= 5
                    self.score -= 5
                    self._set_message("Hazard!", self.COLOR_PENALTY, 30)
                    # Sound: Player hurt
            else:
                self.player_screen_pos[0] += dx / dist * move_speed
                self.player_screen_pos[1] += dy / dist * move_speed

        # --- Interaction (Collecting Clues) ---
        if space_held and not self.last_action_was_interact:
            px, py = self.player_pos
            clue_to_remove = None
            for clue in self.clues:
                cx, cy = clue
                if abs(px - cx) <= 1 and abs(py - cy) <= 1: # Check if adjacent or on
                    clue_to_remove = clue
                    break
            
            if clue_to_remove:
                self.clues.remove(clue_to_remove)
                self.clues_collected_this_stage += 1
                reward += 10
                self.score += 10
                self._set_message("Clue Found!", self.COLOR_CLUE, 60)
                # Sound: Clue collect
                # Particle effect
                clue_screen_pos = self._grid_to_screen(*clue_to_remove)
                for _ in range(20):
                    self.particles.append(self._create_particle(clue_screen_pos, self.COLOR_CLUE))
        self.last_action_was_interact = space_held

        # --- Check Termination/Progression ---
        terminated = False
        if self.time_remaining <= 0:
            reward -= 100
            self.score -= 100
            terminated = True
            self.game_over = True
            self._set_message("Time's Up...", self.COLOR_UI_FAIL, 120)
            # Sound: Game over
        
        if self.clues_collected_this_stage == 5:
            reward += 100
            self.score += 100
            self.current_stage += 1
            if self.current_stage > 3:
                terminated = True
                self.game_over = True
                self._set_message("Mystery Solved!", self.COLOR_UI_SUCCESS, 120)
                # Sound: Victory
            else:
                self._setup_stage() # Progress to next level
                # Sound: Stage complete

        self._update_particles()
        if self.message_timer:
            self.message_timer -= 1
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _create_particle(self, pos, color):
        angle = self.np_random.random() * 2 * math.pi
        speed = self.np_random.random() * 2 + 1
        return {
            "pos": list(pos),
            "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
            "life": self.np_random.integers(20, 40),
            "color": color,
            "size": self.np_random.random() * 3 + 2
        }

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            p["size"] *= 0.98
            if p["life"] > 0 and p["size"] > 0.5:
                active_particles.append(p)
        self.particles = active_particles

    def _set_message(self, text, color, duration):
        self.message_text = (text, color)
        self.message_timer = duration

    def _render_game(self):
        # Render isometric grid and buildings
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                screen_pos = self._grid_to_screen(x, y)
                
                # Tile base
                tile_points = [
                    (screen_pos[0], screen_pos[1] - self.TILE_HEIGHT_HALF),
                    (screen_pos[0] + self.TILE_WIDTH_HALF, screen_pos[1]),
                    (screen_pos[0], screen_pos[1] + self.TILE_HEIGHT_HALF),
                    (screen_pos[0] - self.TILE_WIDTH_HALF, screen_pos[1])
                ]
                pygame.gfxdraw.filled_polygon(self.screen, tile_points, self.COLOR_TILE)
                pygame.gfxdraw.aapolygon(self.screen, tile_points, self.COLOR_TILE_EDGE)

                # Zones
                if (x, y) in self.penalty_zones:
                    pygame.gfxdraw.filled_polygon(self.screen, tile_points, (*self.COLOR_PENALTY, 40))
                
                # Buildings
                if self.world_grid[x, y] == 1:
                    top_y = screen_pos[1] - 40
                    base_points = [
                        (screen_pos[0] - self.TILE_WIDTH_HALF, screen_pos[1]),
                        (screen_pos[0] - self.TILE_WIDTH_HALF, top_y),
                        (screen_pos[0] + self.TILE_WIDTH_HALF, top_y),
                        (screen_pos[0] + self.TILE_WIDTH_HALF, screen_pos[1])
                    ]
                    # Right face
                    pygame.draw.polygon(self.screen, self.COLOR_BUILDING_A, [
                        (screen_pos[0], screen_pos[1] + self.TILE_HEIGHT_HALF),
                        (screen_pos[0] + self.TILE_WIDTH_HALF, screen_pos[1]),
                        (screen_pos[0] + self.TILE_WIDTH_HALF, top_y),
                        (screen_pos[0], top_y + self.TILE_HEIGHT_HALF)
                    ])
                    # Left face
                    pygame.draw.polygon(self.screen, self.COLOR_BUILDING_B, [
                        (screen_pos[0], screen_pos[1] + self.TILE_HEIGHT_HALF),
                        (screen_pos[0] - self.TILE_WIDTH_HALF, screen_pos[1]),
                        (screen_pos[0] - self.TILE_WIDTH_HALF, top_y),
                        (screen_pos[0], top_y + self.TILE_HEIGHT_HALF)
                    ])
                    # Top face
                    pygame.draw.polygon(self.screen, self.COLOR_BUILDING_A, [
                        (screen_pos[0], top_y + self.TILE_HEIGHT_HALF),
                        (screen_pos[0] + self.TILE_WIDTH_HALF, top_y),
                        (screen_pos[0], top_y - self.TILE_HEIGHT_HALF),
                        (screen_pos[0] - self.TILE_WIDTH_HALF, top_y)
                    ])
        
        # Render dynamic objects (sorted by y-pos for correct layering)
        render_queue = []
        # Clues
        for clue_pos in self.clues:
            screen_pos = self._grid_to_screen(*clue_pos)
            render_queue.append(('clue', screen_pos, clue_pos))
        # Player
        render_queue.append(('player', self.player_screen_pos, self.player_pos))
        
        # Sort by grid y, then grid x
        render_queue.sort(key=lambda item: (item[2][1], item[2][0]))

        for item_type, screen_pos, grid_pos in render_queue:
            if item_type == 'clue':
                glow_size = 8 + math.sin(self.steps * 0.1) * 2
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos[0]), int(screen_pos[1]), int(glow_size), (*self.COLOR_CLUE, 60))
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos[0]), int(screen_pos[1]), 4, self.COLOR_CLUE)
                pygame.gfxdraw.aacircle(self.screen, int(screen_pos[0]), int(screen_pos[1]), 4, self.COLOR_CLUE)
            elif item_type == 'player':
                px, py = int(screen_pos[0]), int(screen_pos[1])
                # Player glow
                glow_radius = 15 + math.sin(self.steps * 0.2) * 3
                pygame.gfxdraw.filled_circle(self.screen, px, py - 5, int(glow_radius), (*self.COLOR_PLAYER, 30))
                # Player body
                pygame.draw.circle(self.screen, self.COLOR_PLAYER, (px, py - 5), 6)

        # Render fog and lighting after game world but before UI
        fog_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        fog_surface.fill((*self.COLOR_FOG, 128))

        # Streetlight cutouts
        for light_pos in self.streetlights:
            screen_pos = self._grid_to_screen(*light_pos)
            flicker = 100 + self.np_random.random() * 40
            pygame.draw.circle(fog_surface, (0,0,0,0), screen_pos, int(flicker))

        # Player light cutout
        pygame.draw.circle(fog_surface, (0,0,0,0), (int(self.player_screen_pos[0]), int(self.player_screen_pos[1])), 80)
        
        self.screen.blit(fog_surface, (0,0))
        
        # Render particles on top of everything
        for p in self.particles:
            pygame.draw.circle(self.screen, p["color"], [int(p["pos"][0]), int(p["pos"][1])], max(0, int(p["size"])))

    def _render_ui(self):
        # UI Background panels
        pygame.draw.rect(self.screen, (0, 0, 0, 150), (0, 0, self.WIDTH, 40))
        pygame.draw.rect(self.screen, (0, 0, 0, 100), (0, self.HEIGHT - 25, self.WIDTH, 25))

        # Clue counter
        clue_text = self.font_large.render(f"Clues: {self.clues_collected_this_stage} / 5", True, self.COLOR_UI_TEXT)
        self.screen.blit(clue_text, (15, 7))

        # Timer
        time_sec = self.time_remaining // 30
        time_color = self.COLOR_UI_TEXT if time_sec > 10 else self.COLOR_UI_FAIL
        time_text = self.font_large.render(f"Time: {time_sec:02d}", True, time_color)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 15, 7))

        # Score
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH//2 - score_text.get_width()//2, 10))

        # User Guide
        guide_text = self.font_small.render(self.user_guide, True, self.COLOR_UI_TEXT)
        self.screen.blit(guide_text, (15, self.HEIGHT - 20))
        
        # Message display
        if self.message_timer and self.message_timer > 0:
            text, color = self.message_text
            alpha = min(255, self.message_timer * 10)
            msg_surf = self.font_large.render(text, True, color)
            msg_surf.set_alpha(alpha)
            pos = (self.WIDTH // 2 - msg_surf.get_width() // 2, self.HEIGHT // 2 - msg_surf.get_height() // 2)
            self.screen.blit(msg_surf, pos)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage,
            "time_remaining": self.time_remaining // 30,
            "clues_found": self.clues_collected_this_stage,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # To display the game, we need to create a window
    pygame.display.set_caption("Isometric Horror Explorer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    # --- Manual Control ---
    # 0=none, 1=up, 2=down, 3=left, 4=right
    movement_action = 0 
    # 0=released, 1=held
    space_action = 0
    shift_action = 0
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        keys = pygame.key.get_pressed()
        
        movement_action = 0
        if keys[pygame.K_UP]:
            movement_action = 1
        elif keys[pygame.K_DOWN]:
            movement_action = 2
        elif keys[pygame.K_LEFT]:
            movement_action = 3
        elif keys[pygame.K_RIGHT]:
            movement_action = 4
            
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        # Pygame uses (width, height), numpy uses (height, width)
        # We need to transpose it back for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Info: {info}")
            # Wait a bit before resetting
            pygame.time.wait(3000)
            obs, info = env.reset()

    env.close()