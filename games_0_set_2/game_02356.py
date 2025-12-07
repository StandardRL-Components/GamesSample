
# Generated: 2025-08-27T20:08:02.843380
# Source Brief: brief_02356.md
# Brief Index: 2356

        
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to explore the cavern. "
        "Collect all 20 crystals to win, but watch out for the dark pits!"
    )

    game_description = (
        "Explore procedurally generated isometric caverns, collecting crystals "
        "while avoiding deadly pits to amass a high score."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.WORLD_WIDTH, self.WORLD_HEIGHT = 40, 40
        self.TILE_WIDTH_ISO, self.TILE_HEIGHT_ISO = 48, 24
        self.TILE_DEPTH_ISO = 20
        self.MAX_STEPS = 1000
        self.CRYSTAL_TARGET = 20

        # Colors
        self.COLOR_BG = (26, 28, 44)
        self.COLOR_FLOOR = (60, 64, 92)
        self.COLOR_PIT_TOP = (42, 45, 62)
        self.COLOR_PIT_BOTTOM = (10, 10, 15)
        self.COLOR_WALL_TOP = (80, 88, 123)
        self.COLOR_WALL_SIDE = (68, 74, 107)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_CRYSTAL = (66, 162, 227)
        self.COLOR_CRYSTAL_BONUS = (240, 230, 140)
        self.COLOR_GLOW = (255, 255, 255)
        self.COLOR_TEXT = (230, 230, 240)

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 28)
        self.font_big = pygame.font.Font(None, 72)

        # Game state variables
        self.world_grid = []
        self.player_pos = [0, 0]
        self.crystals = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.animation_tick = 0

        self.reset()
        self.validate_implementation()

    def _generate_cavern(self):
        for _ in range(50): # Retry generation if it fails
            grid = [['wall' for _ in range(self.WORLD_WIDTH)] for _ in range(self.WORLD_HEIGHT)]
            start_pos = [self.WORLD_WIDTH // 2, self.WORLD_HEIGHT // 2]
            
            # 1. Carve cavern using random walk
            px, py = start_pos
            dx, dy = random.choice([(0,1), (0,-1), (1,0), (-1,0)])
            for _ in range(1500):
                grid[py][px] = 'floor'
                if random.random() > 0.8: # Chance to change direction
                    dx, dy = random.choice([(0,1), (0,-1), (1,0), (-1,0)])
                px, py = np.clip(px + dx, 1, self.WORLD_WIDTH - 2), np.clip(py + dy, 1, self.WORLD_HEIGHT - 2)

            # 2. Find all reachable floor tiles via BFS
            q = deque([tuple(start_pos)])
            reachable = {tuple(start_pos)}
            while q:
                x, y = q.popleft()
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.WORLD_WIDTH and 0 <= ny < self.WORLD_HEIGHT and \
                       grid[ny][nx] == 'floor' and (nx, ny) not in reachable:
                        reachable.add((nx, ny))
                        q.append((nx, ny))
            
            # 3. Place pits on edges of the cavern
            pit_locations = set()
            for x, y in list(reachable):
                wall_neighbors = 0
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < self.WORLD_WIDTH and 0 <= ny < self.WORLD_HEIGHT and grid[ny][nx] == 'floor'):
                        wall_neighbors += 1
                if wall_neighbors > 0 and random.random() < 0.2:
                    grid[y][x] = 'pit'
                    pit_locations.add((x, y))

            # 4. Place crystals on valid spots
            valid_crystal_spots = [pos for pos in reachable if grid[pos[1]][pos[0]] == 'floor' and pos != tuple(start_pos)]
            if len(valid_crystal_spots) < self.CRYSTAL_TARGET:
                continue # Regenerate if not enough spots

            random.shuffle(valid_crystal_spots)
            crystal_data = []
            for pos in valid_crystal_spots[:self.CRYSTAL_TARGET]:
                is_bonus = False
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    if (pos[0] + dx, pos[1] + dy) in pit_locations:
                        is_bonus = True
                        break
                crystal_data.append({'pos': list(pos), 'is_bonus': is_bonus})
            
            return grid, start_pos, crystal_data

        raise RuntimeError("Failed to generate a valid cavern.")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.animation_tick = 0
        self.particles = []
        
        self.world_grid, self.player_pos, crystal_data = self._generate_cavern()
        self.crystals = [{'pos': d['pos'], 'is_bonus': d['is_bonus'], 'value': 5 if d['is_bonus'] else 1} for d in crystal_data]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self.animation_tick += 1
        reward = 0
        terminated = False

        # 1. Handle Movement
        px, py = self.player_pos
        moved = False
        if movement == 1: # Up
            py -= 1
            moved = True
        elif movement == 2: # Down
            py += 1
            moved = True
        elif movement == 3: # Left
            px -= 1
            moved = True
        elif movement == 4: # Right
            px += 1
            moved = True

        if moved:
            if 0 <= px < self.WORLD_WIDTH and 0 <= py < self.WORLD_HEIGHT and self.world_grid[py][px] != 'wall':
                self.player_pos = [px, py]
                reward -= 0.1
            else: # Hit a wall, no move
                moved = False

        # 2. Update particle animations
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] - 0.05)
            
        # 3. Check for events at new position
        current_tile = self.world_grid[self.player_pos[1]][self.player_pos[0]]
        
        if current_tile == 'pit':
            reward = -100
            self.game_over = True
            terminated = True
            # sfx: fall_sfx

        else:
            # Check for crystal collection
            collected_crystal = None
            for i, crystal in enumerate(self.crystals):
                if crystal['pos'] == self.player_pos:
                    collected_crystal = self.crystals.pop(i)
                    break
            
            if collected_crystal:
                reward += collected_crystal['value']
                self.score += collected_crystal['value']
                # sfx: crystal_get_sfx
                
                # Spawn particles
                crystal_screen_pos = self._iso_to_screen(self.player_pos[0], self.player_pos[1], [0,0])
                color = self.COLOR_CRYSTAL_BONUS if collected_crystal['is_bonus'] else self.COLOR_CRYSTAL
                for _ in range(15):
                    self.particles.append({
                        'pos': list(crystal_screen_pos),
                        'vel': [random.uniform(-2, 2), random.uniform(-4, -1)],
                        'life': random.randint(20, 40),
                        'radius': random.uniform(2, 5),
                        'color': color
                    })

            # Check for win condition
            if not self.crystals:
                reward += 100
                self.win = True
                self.game_over = True
                terminated = True
                # sfx: win_sfx

        # 4. Check for step limit
        self.steps += 1
        if self.steps >= self.MAX_STEPS and not terminated:
            self.game_over = True
            terminated = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _iso_to_screen(self, x, y, camera_offset):
        screen_x = camera_offset[0] + (x - y) * self.TILE_WIDTH_ISO / 2
        screen_y = camera_offset[1] + (x + y) * self.TILE_HEIGHT_ISO / 2
        return int(screen_x), int(screen_y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Camera follows player
        player_screen_pos = self._iso_to_screen(self.player_pos[0], self.player_pos[1], [0,0])
        camera_offset = [
            self.SCREEN_WIDTH / 2 - player_screen_pos[0],
            self.SCREEN_HEIGHT / 2 - player_screen_pos[1] - self.TILE_DEPTH_ISO
        ]

        # Get all visible objects and sort by depth (y then x)
        draw_list = []
        for y in range(self.WORLD_HEIGHT):
            for x in range(self.WORLD_WIDTH):
                tile_type = self.world_grid[y][x]
                if tile_type != 'wall':
                    draw_list.append({'type': 'tile', 'pos': (x, y), 'tile': tile_type})

        for crystal in self.crystals:
            draw_list.append({'type': 'crystal', 'data': crystal})

        draw_list.append({'type': 'player', 'pos': self.player_pos})
        
        # Sort by isometric depth (y+x)
        draw_list.sort(key=lambda item: (item['pos'][0] + item['pos'][1] if 'pos' in item else item['data']['pos'][0] + item['data']['pos'][1]))

        # Render walls separately from back to front
        for y in range(self.WORLD_HEIGHT):
            for x in range(self.WORLD_WIDTH):
                if self.world_grid[y][x] == 'wall':
                    self._draw_iso_block(x, y, camera_offset)

        # Render sorted items
        for item in draw_list:
            if item['type'] == 'tile':
                self._draw_iso_tile(item['pos'][0], item['pos'][1], item['tile'], camera_offset)
            elif item['type'] == 'crystal':
                self._draw_crystal(item['data'], camera_offset)
            elif item['type'] == 'player':
                self._draw_player(item['pos'], camera_offset)

        # Render particles on top
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(p['radius']))

    def _draw_iso_tile(self, x, y, tile_type, camera_offset):
        screen_pos = self._iso_to_screen(x, y, camera_offset)
        w, h = self.TILE_WIDTH_ISO, self.TILE_HEIGHT_ISO
        points = [
            (screen_pos[0], screen_pos[1]),
            (screen_pos[0] + w / 2, screen_pos[1] + h / 2),
            (screen_pos[0], screen_pos[1] + h),
            (screen_pos[0] - w / 2, screen_pos[1] + h / 2)
        ]
        color = self.COLOR_FLOOR if tile_type == 'floor' else self.COLOR_PIT_TOP
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, color)

        if tile_type == 'pit':
            inner_points = [
                (screen_pos[0], screen_pos[1] + 4),
                (screen_pos[0] + w / 2 - 4, screen_pos[1] + h / 2),
                (screen_pos[0], screen_pos[1] + h - 4),
                (screen_pos[0] - w / 2 + 4, screen_pos[1] + h / 2)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, inner_points, self.COLOR_PIT_BOTTOM)

    def _draw_iso_block(self, x, y, camera_offset):
        screen_pos = self._iso_to_screen(x, y, camera_offset)
        w, h, d = self.TILE_WIDTH_ISO, self.TILE_HEIGHT_ISO, self.TILE_DEPTH_ISO
        
        top_points = [
            (screen_pos[0], screen_pos[1] - d),
            (screen_pos[0] + w / 2, screen_pos[1] + h / 2 - d),
            (screen_pos[0], screen_pos[1] + h - d),
            (screen_pos[0] - w / 2, screen_pos[1] + h / 2 - d)
        ]
        
        # Draw front-left face
        if y + 1 < self.WORLD_HEIGHT and self.world_grid[y + 1][x] != 'wall':
            side1_points = [
                (screen_pos[0] - w / 2, screen_pos[1] + h / 2 - d),
                (screen_pos[0], screen_pos[1] + h - d),
                (screen_pos[0], screen_pos[1] + h),
                (screen_pos[0] - w / 2, screen_pos[1] + h / 2)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, side1_points, self.COLOR_WALL_SIDE)
        
        # Draw front-right face
        if x + 1 < self.WORLD_WIDTH and self.world_grid[y][x + 1] != 'wall':
            side2_points = [
                (screen_pos[0] + w / 2, screen_pos[1] + h / 2 - d),
                (screen_pos[0], screen_pos[1] + h - d),
                (screen_pos[0], screen_pos[1] + h),
                (screen_pos[0] + w / 2, screen_pos[1] + h / 2)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, side2_points, self.COLOR_WALL_SIDE)
            
        pygame.gfxdraw.filled_polygon(self.screen, top_points, self.COLOR_WALL_TOP)
        pygame.gfxdraw.aapolygon(self.screen, top_points, self.COLOR_WALL_TOP)
        
    def _draw_crystal(self, crystal, camera_offset):
        x, y = crystal['pos']
        screen_pos = self._iso_to_screen(x, y, camera_offset)
        
        # Desynchronize animation per crystal
        anim_offset = (x * 5 + y * 3)
        bob = math.sin((self.animation_tick + anim_offset) * 0.05) * 4
        
        color = self.COLOR_CRYSTAL_BONUS if crystal['is_bonus'] else self.COLOR_CRYSTAL
        w, h = 12, 24
        
        if crystal['is_bonus']:
            glow_radius = 12 + math.sin((self.animation_tick + anim_offset) * 0.08) * 4
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_GLOW, 60), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (screen_pos[0] - glow_radius, screen_pos[1] - glow_radius - h / 2 - bob))

        points = [
            (screen_pos[0], screen_pos[1] - h / 2 - bob),
            (screen_pos[0] + w / 2, screen_pos[1] - bob),
            (screen_pos[0], screen_pos[1] + h / 2 - bob),
            (screen_pos[0] - w / 2, screen_pos[1] - bob),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _draw_player(self, pos, camera_offset):
        screen_pos = self._iso_to_screen(pos[0], pos[1], camera_offset)
        bob = math.sin(self.animation_tick * 0.15) * 4
        h = self.TILE_HEIGHT_ISO
        
        # Shadow
        shadow_points = [
            (screen_pos[0], screen_pos[1] + h/2 + 2),
            (screen_pos[0] + 8, screen_pos[1] + h/2 + 2 + 4),
            (screen_pos[0], screen_pos[1] + h/2 + 2 + 8),
            (screen_pos[0] - 8, screen_pos[1] + h/2 + 2 + 4)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, shadow_points, (0,0,0,50))

        # Player triangle
        points = [
            (screen_pos[0], screen_pos[1] - h/2 - bob),
            (screen_pos[0] - 10, screen_pos[1] + h/2 - bob),
            (screen_pos[0] + 10, screen_pos[1] + h/2 - bob),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        crystals_text = self.font_main.render(f"Crystals Left: {len(self.crystals)}", True, self.COLOR_TEXT)
        self.screen.blit(crystals_text, (10, 40))

        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (152, 251, 152) if self.win else (255, 100, 100)
            end_text = self.font_big.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "crystals_left": len(self.crystals)}

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # To play manually, you need a window.
    # This part is for demonstration and will not run in a headless environment.
    try:
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Isometric Crystal Caverns")
        
        obs, info = env.reset()
        done = False
        
        print(env.user_guide)

        while not done:
            action = [0, 0, 0] # Default action: no-op
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            # Since auto_advance is False, we only step when a key is pressed.
            # For a better human play experience, we can step on every frame.
            # This is a small deviation from the strict turn-based logic for playability.
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Draw the observation to the display window
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(15) # Limit frame rate for human play

            if done:
                print(f"Game Over! Final Info: {info}")
                pygame.time.wait(3000) # Pause before reset
                obs, info = env.reset()
                done = False

    except Exception as e:
        print(f"An error occurred during manual play: {e}")
        print("Note: Manual play requires a display. The environment is designed for headless operation.")
    finally:
        env.close()