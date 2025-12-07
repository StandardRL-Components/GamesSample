
# Generated: 2025-08-28T04:43:39.753503
# Source Brief: brief_05342.md
# Brief Index: 5342

        
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

    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move your avatar on the grid. "
        "Avoid red traps and collect blue crystals."
    )

    game_description = (
        "A strategic puzzle game. Navigate an isometric grid to collect 20 blue crystals "
        "while avoiding hazardous red traps. You lose if you hit 3 traps."
    )

    auto_advance = False

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 12, 12
    TILE_W, TILE_H = 40, 20
    ORIGIN_X, ORIGIN_Y = WIDTH // 2, 80

    NUM_CRYSTALS_START = 30
    NUM_TRAPS = 15
    WIN_CRYSTALS = 20
    LOSE_TRAPS = 3
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (45, 50, 62)
    COLOR_PLAYER = (43, 255, 136)
    COLOR_PLAYER_SIDE = (30, 180, 95)
    COLOR_CRYSTAL = (0, 191, 255)
    COLOR_CRYSTAL_SIDE = (0, 130, 175)
    COLOR_CRYSTAL_GLOW = (120, 220, 255)
    COLOR_TRAP = (255, 50, 50)
    COLOR_TRAP_SIDE = (180, 35, 35)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    COLOR_WIN = (173, 255, 47)
    COLOR_LOSE = (255, 69, 0)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.ui_font = pygame.font.Font(None, 28)
        self.end_font = pygame.font.Font(None, 72)

        self.player_pos = [0, 0]
        self.crystals = []
        self.traps = []
        self.particles = []
        self.steps = 0
        self.crystal_score = 0
        self.trap_hits = 0
        self.total_reward = 0.0
        self.game_over = False
        self.win_message = ""
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.crystal_score = 0
        self.trap_hits = 0
        self.total_reward = 0.0
        self.game_over = False
        self.win_message = ""
        self.particles.clear()
        
        self._place_entities()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        
        old_player_pos = list(self.player_pos)
        
        # --- Player Movement ---
        if movement == 1: # Up
            self.player_pos[1] -= 1
        elif movement == 2: # Down
            self.player_pos[1] += 1
        elif movement == 3: # Left
            self.player_pos[0] -= 1
        elif movement == 4: # Right
            self.player_pos[0] += 1
        
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.GRID_WIDTH - 1)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.GRID_HEIGHT - 1)
        
        # --- Reward Calculation ---
        # Proximity rewards
        dist_crystal_before = self._find_closest_distance(old_player_pos, self.crystals)
        dist_crystal_after = self._find_closest_distance(self.player_pos, self.crystals)
        if dist_crystal_after < dist_crystal_before:
            reward += 1.0

        dist_trap_before = self._find_closest_distance(old_player_pos, self.traps)
        dist_trap_after = self._find_closest_distance(self.player_pos, self.traps)
        if dist_trap_after < dist_trap_before:
            reward -= 0.1
            
        # --- Collision & Event Rewards ---
        crystal_collected = None
        for crystal in self.crystals:
            if self.player_pos == crystal:
                crystal_collected = crystal
                break
        
        if crystal_collected:
            self.crystals.remove(crystal_collected)
            self.crystal_score += 1
            reward += 10.0
            # sfx: crystal_get.wav
            self._spawn_particles(self.player_pos, self.COLOR_CRYSTAL, 20)
        
        if self.player_pos in self.traps:
            self.trap_hits += 1
            reward -= 5.0
            # sfx: trap_hit.wav
            self._spawn_particles(self.player_pos, self.COLOR_TRAP, 30, speed_mult=1.5)
            # To avoid repeated penalties, we move the player back if they land on a trap
            self.player_pos = old_player_pos

        self.total_reward += reward
        self.steps += 1
        
        # --- Update Particles ---
        self._update_particles()
        
        # --- Termination Check ---
        terminated = False
        if self.crystal_score >= self.WIN_CRYSTALS:
            terminated = True
            self.game_over = True
            reward += 100.0
            self.win_message = "VICTORY!"
            # sfx: game_win.wav
        elif self.trap_hits >= self.LOSE_TRAPS:
            terminated = True
            self.game_over = True
            reward -= 100.0
            self.win_message = "DEFEAT"
            # sfx: game_lose.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.win_message = "TIME UP"

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _place_entities(self):
        all_coords = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        
        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        available_coords = [c for c in all_coords if list(c) != self.player_pos]
        self.np_random.shuffle(available_coords)
        
        # Place traps
        self.traps = [list(c) for c in available_coords[:self.NUM_TRAPS]]
        
        # Place crystals
        available_coords = available_coords[self.NUM_TRAPS:]
        self.crystals = [list(c) for c in available_coords[:self.NUM_CRYSTALS_START]]

        # Ensure a safe path to at least one crystal
        player_neighbors = [
            (self.player_pos[0] + dx, self.player_pos[1] + dy)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
        ]
        safe_neighbors = [
            n for n in player_neighbors
            if 0 <= n[0] < self.GRID_WIDTH and 0 <= n[1] < self.GRID_HEIGHT and list(n) not in self.traps
        ]
        
        has_neighbor_crystal = any(list(n) in self.crystals for n in safe_neighbors)
        
        if not has_neighbor_crystal and safe_neighbors and self.crystals:
            # If no crystal is nearby, move one to a safe neighbor spot
            crystal_to_move = self.crystals.pop(0)
            new_pos = list(self.np_random.choice(safe_neighbors))
            # Make sure we don't place it on another crystal
            while new_pos in self.crystals:
                safe_neighbors.remove(tuple(new_pos))
                if not safe_neighbors: break # Should not happen
                new_pos = list(self.np_random.choice(safe_neighbors))
            
            if new_pos not in self.crystals:
                 self.crystals.append(new_pos)

    def _find_closest_distance(self, pos, entity_list):
        if not entity_list:
            return float('inf')
        min_dist = float('inf')
        for entity in entity_list:
            dist = abs(pos[0] - entity[0]) + abs(pos[1] - entity[1]) # Manhattan distance
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _iso_to_screen(self, gx, gy):
        sx = self.ORIGIN_X + (gx - gy) * self.TILE_W / 2
        sy = self.ORIGIN_Y + (gx + gy) * self.TILE_H / 2
        return int(sx), int(sy)

    def _draw_iso_cube(self, surface, color, side_color, grid_pos, height_mod=0):
        x, y = grid_pos
        sx, sy = self._iso_to_screen(x, y)
        sy -= int(height_mod)

        points = [
            (sx, sy - self.TILE_H // 2),
            (sx + self.TILE_W // 2, sy),
            (sx, sy + self.TILE_H // 2),
            (sx - self.TILE_W // 2, sy),
        ]
        
        bottom_y = sy + self.TILE_H
        
        # Right side face
        pygame.gfxdraw.filled_polygon(surface, [
            (sx + self.TILE_W // 2, sy),
            (sx, sy + self.TILE_H // 2),
            (sx, bottom_y + self.TILE_H // 2),
            (sx + self.TILE_W // 2, bottom_y),
        ], side_color)
        
        # Left side face
        pygame.gfxdraw.filled_polygon(surface, [
            (sx - self.TILE_W // 2, sy),
            (sx, sy + self.TILE_H // 2),
            (sx, bottom_y + self.TILE_H // 2),
            (sx - self.TILE_W // 2, bottom_y),
        ], side_color)
        
        # Top face
        pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, color)

    def _spawn_particles(self, grid_pos, color, count, speed_mult=1.0):
        sx, sy = self._iso_to_screen(grid_pos[0], grid_pos[1])
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed - speed * 0.5]
            self.particles.append({
                'pos': [sx, sy],
                'vel': vel,
                'lifetime': self.np_random.integers(20, 40),
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['lifetime'] -= 1
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # Render grid
        for y in range(self.GRID_HEIGHT + 1):
            start = self._iso_to_screen(0, y)
            end = self._iso_to_screen(self.GRID_WIDTH, y)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for x in range(self.GRID_WIDTH + 1):
            start = self._iso_to_screen(x, 0)
            end = self._iso_to_screen(x, self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

        # Create a sorted list of all entities for correct isometric rendering
        render_list = []
        for trap_pos in self.traps:
            render_list.append({'type': 'trap', 'pos': trap_pos})
        for crystal_pos in self.crystals:
            render_list.append({'type': 'crystal', 'pos': crystal_pos})
        render_list.append({'type': 'player', 'pos': self.player_pos})
        
        render_list.sort(key=lambda e: (e['pos'][0] + e['pos'][1], e['pos'][1]))

        # Render entities
        for item in render_list:
            pos = item['pos']
            if item['type'] == 'trap':
                self._draw_iso_cube(self.screen, self.COLOR_TRAP, self.COLOR_TRAP_SIDE, pos)
            elif item['type'] == 'crystal':
                # Pulsing glow effect
                glow_alpha = (math.sin(self.steps * 0.15) * 0.4 + 0.6) * 100
                sx, sy = self._iso_to_screen(pos[0], pos[1])
                glow_radius = int(self.TILE_W * 0.6)
                s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*self.COLOR_CRYSTAL_GLOW, glow_alpha), (glow_radius, glow_radius), glow_radius)
                self.screen.blit(s, (sx - glow_radius, sy - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
                self._draw_iso_cube(self.screen, self.COLOR_CRYSTAL, self.COLOR_CRYSTAL_SIDE, pos)
            elif item['type'] == 'player':
                bob = math.sin(self.steps * 0.2) * 3
                self._draw_iso_cube(self.screen, self.COLOR_PLAYER, self.COLOR_PLAYER_SIDE, pos, height_mod=bob)

        # Render particles
        for p in self.particles:
            size = max(0, p['size'] * (p['lifetime'] / 40.0))
            pygame.draw.circle(self.screen, p['color'], p['pos'], size)
            
        # Render UI
        crystal_text = f"Crystals: {self.crystal_score} / {self.WIN_CRYSTALS}"
        trap_text = f"Traps Hit: {self.trap_hits} / {self.LOSE_TRAPS}"
        
        self._render_text(crystal_text, (10, 10))
        self._render_text(trap_text, (10, 35))
        
        if self.game_over:
            color = self.COLOR_WIN if self.win_message == "VICTORY!" else self.COLOR_LOSE
            self._render_text(self.win_message, (self.WIDTH//2, self.HEIGHT//2), font=self.end_font, color=color, center=True)
            
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_text(self, text, pos, font=None, color=None, center=False):
        if font is None: font = self.ui_font
        if color is None: color = self.COLOR_TEXT
        
        shadow_surface = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_surface = font.render(text, True, color)
        
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
            
        self.screen.blit(shadow_surface, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.total_reward,
            "steps": self.steps,
            "crystals_collected": self.crystal_score,
            "traps_hit": self.trap_hits,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
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
    
    # --- Manual Play Example ---
    # This requires setting up a pygame window to display the frames
    
    obs, info = env.reset()
    done = False
    
    # Setup a window to display the game
    pygame.display.set_caption("Isometric Crystal Collector")
    display_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    action = [0, 0, 0] # No-op
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # --- Map keyboard to MultiDiscrete action space ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # Since auto_advance is False, we only step when there's a movement action
        if movement != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Total: {info['score']:.2f}, Done: {done}")

        # --- Render the observation to the display window ---
        # The observation is (H, W, C), but pygame blit needs (W, H) surface
        # We can just get the internal surface from the env
        display_screen.blit(env.screen, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(15) # Limit frame rate for manual play
        
    env.close()
    print("Game Over!")