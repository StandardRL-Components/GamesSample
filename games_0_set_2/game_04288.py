
# Generated: 2025-08-28T01:56:58.016832
# Source Brief: brief_04288.md
# Brief Index: 4288

        
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


# Helper class for floating text particles (e.g., damage numbers)
class Particle:
    def __init__(self, text, pos, color, lifetime=30, velocity=(0, -0.5)):
        self.text = text
        self.pos = list(pos)
        self.color = color
        self.lifetime = lifetime
        self.velocity = velocity
        self.life = 0

    def update(self):
        self.pos[0] += self.velocity[0]
        self.pos[1] += self.velocity[1]
        self.life += 1
        return self.life >= self.lifetime

# Main Game Environment
class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Hold Space and press an arrow key to attack in that direction."
    )
    game_description = (
        "Clear the arena by defeating all 7 monsters in this turn-based tactical grid combat game."
    )

    auto_advance = False

    # --- Constants ---
    GRID_SIZE = (16, 10)
    TILE_SIZE = 40
    SCREEN_WIDTH = GRID_SIZE[0] * TILE_SIZE
    SCREEN_HEIGHT = GRID_SIZE[1] * TILE_SIZE

    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    
    COLOR_PLAYER = (50, 255, 150)
    COLOR_PLAYER_ACCENT = (200, 255, 220)
    COLOR_MONSTER = (255, 80, 80)
    COLOR_MONSTER_ACCENT = (255, 180, 180)
    COLOR_HEALTH_GREEN = (0, 200, 0)
    COLOR_HEALTH_RED = (200, 0, 0)
    COLOR_HEALTH_BG = (50, 50, 50)
    
    COLOR_TEXT = (240, 240, 240)
    COLOR_DAMAGE_PLAYER = (255, 100, 100)
    COLOR_DAMAGE_MONSTER = (255, 255, 200)

    PLAYER_MAX_HEALTH = 100
    MONSTER_MAX_HEALTH = 20
    MONSTER_COUNT = 7
    PLAYER_ATTACK_DAMAGE = 10
    MONSTER_ATTACK_DAMAGE = 5
    MAX_STEPS = 1000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_particle = pygame.font.SysFont("monospace", 16, bold=True)
        
        self.player = {}
        self.monsters = []
        self.particles = []
        self.attack_effect = None
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.attack_effect = None

        occupied_positions = set()

        player_start_pos = (self.np_random.integers(0, self.GRID_SIZE[0]), self.np_random.integers(0, self.GRID_SIZE[1]))
        self.player = {
            "pos": list(player_start_pos),
            "health": self.PLAYER_MAX_HEALTH,
            "facing": 1, # 1:up, 2:down, 3:left, 4:right
            "is_hit": False,
        }
        occupied_positions.add(tuple(self.player["pos"]))

        self.monsters = []
        for _ in range(self.MONSTER_COUNT):
            while True:
                pos = (self.np_random.integers(0, self.GRID_SIZE[0]), self.np_random.integers(0, self.GRID_SIZE[1]))
                if pos not in occupied_positions:
                    self.monsters.append({
                        "pos": list(pos),
                        "health": self.MONSTER_MAX_HEALTH,
                        "is_alive": True,
                        "is_hit": False
                    })
                    occupied_positions.add(pos)
                    break
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Reset turn-based flags
        self.player["is_hit"] = False
        for monster in self.monsters:
            monster["is_hit"] = False
        self.attack_effect = None

        reward = -0.1  # Cost of living
        self.steps += 1

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Player Action Phase ---
        is_attack_action = space_held and movement > 0
        is_move_action = not space_held and movement > 0

        if is_attack_action:
            self.player["facing"] = movement
            target_pos = self._get_neighbor_pos(self.player["pos"], movement)
            
            # Visual effect for the attack
            self.attack_effect = {
                "start": self._grid_to_pixel_center(self.player["pos"]),
                "end": self._grid_to_pixel_center(target_pos),
                "life": 5 # Arbitrary short life for visual effect
            }

            for monster in self.monsters:
                if monster["is_alive"] and tuple(monster["pos"]) == target_pos:
                    monster["health"] -= self.PLAYER_ATTACK_DAMAGE
                    monster["is_hit"] = True
                    reward += 1.0
                    self._create_particle(f"-{self.PLAYER_ATTACK_DAMAGE}", monster["pos"], self.COLOR_DAMAGE_MONSTER)
                    if monster["health"] <= 0:
                        monster["is_alive"] = False
                    break
        
        elif is_move_action:
            self.player["facing"] = movement
            target_pos = self._get_neighbor_pos(self.player["pos"], movement)
            
            if self._is_valid_and_empty(target_pos):
                self.player["pos"] = list(target_pos)

        # --- Monster Action Phase ---
        monster_positions_before_move = {i: tuple(m["pos"]) for i, m in enumerate(self.monsters) if m["is_alive"]}
        
        for i, monster in enumerate(self.monsters):
            if not monster["is_alive"]:
                continue

            # Check for contact before monster moves (if player moved into them)
            if tuple(monster["pos"]) == tuple(self.player["pos"]):
                 self.player["health"] -= self.MONSTER_ATTACK_DAMAGE
                 self.player["is_hit"] = True
                 reward -= 1.0
                 self._create_particle(f"-{self.MONSTER_ATTACK_DAMAGE}", self.player["pos"], self.COLOR_DAMAGE_PLAYER)
                 # Push player back to previous valid spot (not perfectly realistic, but prevents overlap)
                 self.player["pos"] = list(self._get_neighbor_pos(monster["pos"], self._get_opposite_dir(self.player["facing"])))


            # Monster moves randomly
            move_dir = self.np_random.integers(1, 5) # 1-4
            target_pos = self._get_neighbor_pos(monster["pos"], move_dir)
            
            if self._is_valid_and_empty(target_pos, ignored_monster_idx=i):
                monster["pos"] = list(target_pos)

            # Check for contact after monster moves
            if tuple(monster["pos"]) == tuple(self.player["pos"]):
                 self.player["health"] -= self.MONSTER_ATTACK_DAMAGE
                 self.player["is_hit"] = True
                 reward -= 1.0
                 self._create_particle(f"-{self.MONSTER_ATTACK_DAMAGE}", self.player["pos"], self.COLOR_DAMAGE_PLAYER)
                 # Push monster back
                 monster["pos"] = list(monster_positions_before_move[i])

        # --- Update and Termination Phase ---
        self.score += reward
        
        # Update particles
        self.particles = [p for p in self.particles if not p.update()]

        # Check for termination
        terminated = False
        monsters_alive = sum(1 for m in self.monsters if m["is_alive"])
        
        if self.player["health"] <= 0:
            self.player["health"] = 0
            reward -= 100
            terminated = True
            self.game_over = True
        elif monsters_alive == 0:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        self.score += reward if terminated else 0

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw monsters
        for monster in self.monsters:
            if monster["is_alive"]:
                self._draw_entity(monster, self.COLOR_MONSTER, self.COLOR_MONSTER_ACCENT, self.MONSTER_MAX_HEALTH)

        # Draw player
        self._draw_entity(self.player, self.COLOR_PLAYER, self.COLOR_PLAYER_ACCENT, self.PLAYER_MAX_HEALTH, is_player=True)

        # Draw attack effect
        if self.attack_effect:
            pygame.draw.line(self.screen, self.COLOR_PLAYER_ACCENT, self.attack_effect["start"], self.attack_effect["end"], 4)
            self.attack_effect["life"] -= 1
            if self.attack_effect["life"] <= 0:
                self.attack_effect = None
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 - (p.life / p.lifetime) * 255)
            text_surf = self.font_particle.render(p.text, True, p.color)
            text_surf.set_alpha(alpha)
            self.screen.blit(text_surf, (int(p.pos[0]), int(p.pos[1])))

    def _draw_entity(self, entity, color, accent_color, max_health, is_player=False):
        pos_px = self._grid_to_pixel_center(entity["pos"])
        radius = int(self.TILE_SIZE * 0.4)
        
        # Hit flash
        draw_color = (255, 255, 255) if entity.get("is_hit", False) else color

        # Body
        pygame.gfxdraw.filled_circle(self.screen, pos_px[0], pos_px[1], radius, draw_color)
        pygame.gfxdraw.aacircle(self.screen, pos_px[0], pos_px[1], radius, draw_color)
        
        # Accent / Facing indicator
        if is_player:
            facing_vec = self._dir_to_vec(entity["facing"])
            eye_pos = (pos_px[0] + facing_vec[0] * radius * 0.6, pos_px[1] + facing_vec[1] * radius * 0.6)
            pygame.gfxdraw.filled_circle(self.screen, int(eye_pos[0]), int(eye_pos[1]), int(radius * 0.3), accent_color)
        else:
            pygame.gfxdraw.filled_circle(self.screen, pos_px[0], pos_px[1], int(radius * 0.5), accent_color)

        # Health bar
        if entity["health"] < max_health:
            bar_pos = (pos_px[0] - radius, pos_px[1] - radius - 8)
            health_pct = entity["health"] / max_health
            self._draw_health_bar(bar_pos, (radius * 2, 5), health_pct, self.COLOR_HEALTH_GREEN, self.COLOR_HEALTH_RED)

    def _render_ui(self):
        # Player Health
        health_text = self.font_ui.render("HEALTH", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 10))
        self._draw_health_bar((80, 14), (150, 12), self.player["health"] / self.PLAYER_MAX_HEALTH, self.COLOR_HEALTH_GREEN, self.COLOR_HEALTH_RED)

        # Monsters Remaining
        monsters_alive = sum(1 for m in self.monsters if m["is_alive"])
        monster_text = self.font_ui.render(f"MONSTERS: {monsters_alive}/{self.MONSTER_COUNT}", True, self.COLOR_TEXT)
        text_rect = monster_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(monster_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player["health"],
            "monsters_left": sum(1 for m in self.monsters if m["is_alive"]),
        }

    # --- Helper Functions ---
    def _grid_to_pixel_center(self, grid_pos):
        x = int((grid_pos[0] + 0.5) * self.TILE_SIZE)
        y = int((grid_pos[1] + 0.5) * self.TILE_SIZE)
        return x, y

    def _get_neighbor_pos(self, pos, direction):
        dx, dy = self._dir_to_vec(direction)
        return (pos[0] + dx, pos[1] + dy)
    
    def _dir_to_vec(self, direction):
        if direction == 1: return (0, -1)  # Up
        if direction == 2: return (0, 1)   # Down
        if direction == 3: return (-1, 0)  # Left
        if direction == 4: return (1, 0)   # Right
        return (0, 0)
    
    def _get_opposite_dir(self, direction):
        if direction == 1: return 2
        if direction == 2: return 1
        if direction == 3: return 4
        if direction == 4: return 3
        return 0

    def _is_valid_and_empty(self, pos, ignored_monster_idx=None):
        # Check boundaries
        if not (0 <= pos[0] < self.GRID_SIZE[0] and 0 <= pos[1] < self.GRID_SIZE[1]):
            return False
        # Check against player
        if tuple(self.player["pos"]) == pos:
            return False
        # Check against monsters
        for i, monster in enumerate(self.monsters):
            if ignored_monster_idx is not None and i == ignored_monster_idx:
                continue
            if monster["is_alive"] and tuple(monster["pos"]) == pos:
                return False
        return True

    def _draw_health_bar(self, pos, size, pct, color_full, color_empty):
        pct = max(0, min(1, pct))
        bg_rect = pygame.Rect(pos, size)
        fill_rect = pygame.Rect(pos, (int(size[0] * pct), size[1]))
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, bg_rect)
        if pct > 0:
            lerp_color = [c1 * pct + c2 * (1-pct) for c1, c2 in zip(color_full, color_empty)]
            pygame.draw.rect(self.screen, lerp_color, fill_rect)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, bg_rect, 1)

    def _create_particle(self, text, grid_pos, color):
        pixel_pos = self._grid_to_pixel_center(grid_pos)
        centered_pos = (pixel_pos[0] - self.font_particle.size(text)[0] / 2, pixel_pos[1] - 20)
        self.particles.append(Particle(text, centered_pos, color))

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
        assert trunc == False
        assert isinstance(info, dict)
        
        # Test specific mechanics
        initial_monsters = sum(1 for m in self.monsters if m['is_alive'])
        assert initial_monsters == self.MONSTER_COUNT, f"Expected {self.MONSTER_COUNT} monsters, found {initial_monsters}"
        assert self.player['health'] <= self.PLAYER_MAX_HEALTH
        
        print("✓ Implementation validated successfully")