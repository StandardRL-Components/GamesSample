import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from typing import List, Tuple, Optional
import os
import os
import pygame


# Set headless mode for pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper classes for game entities and effects
class Unit:
    """Base class for player and monster units."""
    def __init__(self, x: int, y: int, hp: int, color: Tuple[int, int, int]):
        self.x = x
        self.y = y
        self.max_hp = hp
        self.hp = hp
        self.color = color
        self.is_alive = True
        self.render_pos = (x, y) # For smooth animation, though not used with auto_advance=False

    def take_damage(self, amount: int):
        self.hp = max(0, self.hp - amount)
        if self.hp == 0:
            self.is_alive = False
        return amount

    def draw(self, surface: pygame.Surface, cell_size: int):
        raise NotImplementedError

    def draw_health_bar(self, surface: pygame.Surface, cell_size: int):
        bar_width = cell_size * 0.8
        bar_height = 5
        x_pos = self.x * cell_size + (cell_size - bar_width) / 2
        y_pos = self.y * cell_size - bar_height - 2

        health_ratio = self.hp / self.max_hp
        
        pygame.draw.rect(surface, (50, 50, 50), (x_pos, y_pos, bar_width, bar_height))
        pygame.draw.rect(surface, self.color, (x_pos, y_pos, bar_width * health_ratio, bar_height))

class PlayerUnit(Unit):
    """Player-controlled hero unit."""
    def __init__(self, x: int, y: int):
        super().__init__(x, y, hp=10, color=(0, 255, 128))
        self.has_acted = False

    def draw(self, surface: pygame.Surface, cell_size: int):
        center_x = int(self.x * cell_size + cell_size / 2)
        center_y = int(self.y * cell_size + cell_size / 2)
        radius = int(cell_size * 0.35)
        
        points = [
            (center_x, center_y - radius),
            (center_x - radius, center_y + radius),
            (center_x + radius, center_y + radius),
        ]
        pygame.gfxdraw.aapolygon(surface, points, self.color)
        pygame.gfxdraw.filled_polygon(surface, points, self.color)

class MonsterUnit(Unit):
    """AI-controlled monster unit."""
    def __init__(self, x: int, y: int):
        super().__init__(x, y, hp=5, color=(255, 80, 80))

    def draw(self, surface: pygame.Surface, cell_size: int):
        center_x = int(self.x * cell_size + cell_size / 2)
        center_y = int(self.y * cell_size + cell_size / 2)
        size = int(cell_size * 0.7)
        rect = pygame.Rect(center_x - size/2, center_y - size/2, size, size)
        pygame.draw.rect(surface, self.color, rect, border_radius=3)

class Animation:
    """For visual effects like particles and flashes."""
    def __init__(self, x: int, y: int, lifetime: int, color: Tuple[int, int, int], radius: float):
        self.x = x
        self.y = y
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.color = color
        self.radius = radius

    def update(self):
        self.lifetime -= 1
        return self.lifetime > 0

    def draw(self, surface: pygame.Surface):
        progress = self.lifetime / self.max_lifetime
        current_radius = int(self.radius * (1 - progress))
        alpha = int(255 * progress)
        if alpha > 0 and current_radius > 0:
            temp_surface = pygame.Surface((current_radius * 2, current_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surface, (*self.color, alpha), (current_radius, current_radius), current_radius)
            surface.blit(temp_surface, (self.x - current_radius, self.y - current_radius))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move selected unit. Space to attack. Shift to cycle units. No-op (0,0,0) to end turn."
    )

    game_description = (
        "Command a squad of heroes on a grid to strategically defeat waves of monsters in a turn-based tactical game."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and grid dimensions to match 640x400 observation space
        self.GRID_SIZE = (16, 10)
        self.CELL_SIZE = 40
        self.SCREEN_WIDTH = self.GRID_SIZE[0] * self.CELL_SIZE  # 16 * 40 = 640
        self.SCREEN_HEIGHT = self.GRID_SIZE[1] * self.CELL_SIZE # 10 * 40 = 400
        
        # Spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("Consolas", 18)
        self.font_m = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Colors
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_SELECT = (60, 120, 255, 100)
        self.COLOR_MOVE = (60, 180, 255, 60)
        self.COLOR_ATTACK = (255, 80, 80, 80)
        self.COLOR_UI = (220, 220, 240)
        
        # State variables (initialized in reset)
        self.steps: int = 0
        self.score: float = 0
        self.game_over: bool = False
        self.wave: int = 0
        self.player_units: List[PlayerUnit] = []
        self.monster_units: List[MonsterUnit] = []
        self.animations: List[Animation] = []
        self.selected_unit_idx: int = 0
        self.rng = np.random.default_rng()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave = 0
        self.animations = []
        
        self._spawn_players()
        self._start_next_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        self.steps += 1

        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()
        
        self._handle_player_action(action)
        
        # End turn if no-op action is given or all players have acted
        is_no_op = action[0] == 0 and action[1] == 0 and action[2] == 0
        all_players_acted = all(u.has_acted for u in self.player_units if u.is_alive)
        
        if is_no_op or all_players_acted:
            # Player explicitly or implicitly ends turn, trigger monster phase
            reward += self._handle_monster_turn()

            # Reset player units for the next turn
            for unit in self.player_units:
                unit.has_acted = False
            self._cycle_selection(forward=True, force_first=True) # Reset selection to first unit

        # Process state changes
        damage_dealt_reward = self._cleanup_dead_units()
        reward += damage_dealt_reward

        # Check for end conditions
        if not self.player_units:
            self.game_over = True
            reward -= 10.0  # Lose penalty
        elif not self.monster_units:
            reward += 10.0  # Wave clear bonus
            self._start_next_wave()

        terminated = self.game_over
        truncated = self.steps >= 1000
        if truncated:
            terminated = True # Per Gymnasium API, if truncated, terminated should also be true
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _spawn_players(self):
        self.player_units = [
            PlayerUnit(1, self.GRID_SIZE[1] // 2 - 1),
            PlayerUnit(1, self.GRID_SIZE[1] // 2),
            PlayerUnit(1, self.GRID_SIZE[1] // 2 + 1),
        ]

    def _start_next_wave(self):
        self.wave += 1
        for unit in self.player_units:
            unit.hp = unit.max_hp # Heal players
            unit.has_acted = False

        self.monster_units = []
        num_monsters = 2 + self.wave
        occupied_coords = {(u.x, u.y) for u in self.player_units}
        
        for _ in range(num_monsters):
            while True:
                x = self.rng.integers(self.GRID_SIZE[0] // 2, self.GRID_SIZE[0])
                y = self.rng.integers(0, self.GRID_SIZE[1])
                if (x, y) not in occupied_coords:
                    self.monster_units.append(MonsterUnit(x, y))
                    occupied_coords.add((x, y))
                    break
        self._cycle_selection(forward=True, force_first=True)

    def _handle_player_action(self, action: np.ndarray) -> None:
        if not self.player_units:
            return

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if shift_held:
            self._cycle_selection(forward=True)
            return

        selected_unit = self.player_units[self.selected_unit_idx]
        if selected_unit.has_acted:
            return

        if space_held:
            # Attack
            target = self._find_adjacent_target(selected_unit, self.monster_units)
            if target:
                target.take_damage(3) # Player deals 3 damage
                self.animations.append(Animation(target.x * self.CELL_SIZE + self.CELL_SIZE//2, target.y * self.CELL_SIZE + self.CELL_SIZE//2, 10, target.color, 20))
                selected_unit.has_acted = True
        elif movement > 0:
            # Move
            dx, dy = [(0,0), (0,-1), (0,1), (-1,0), (1,0)][movement]
            nx, ny = selected_unit.x + dx, selected_unit.y + dy
            if self._is_valid_move(nx, ny):
                selected_unit.x, selected_unit.y = nx, ny
                selected_unit.has_acted = True

    def _handle_monster_turn(self) -> float:
        reward = 0.0
        all_units_coords = {(u.x, u.y) for u in self.player_units + self.monster_units}

        for monster in self.monster_units:
            if not monster.is_alive or not self.player_units:
                continue

            # 1. Attack if possible
            target = self._find_adjacent_target(monster, self.player_units)
            if target:
                damage = target.take_damage(2) # Monster deals 2 damage
                reward -= 0.1 * damage
                self.animations.append(Animation(target.x * self.CELL_SIZE + self.CELL_SIZE//2, target.y * self.CELL_SIZE + self.CELL_SIZE//2, 10, target.color, 15))
            else:
                # 2. Move if not attacking
                # Find nearest player
                nearest_player = min(self.player_units, key=lambda p: abs(p.x - monster.x) + abs(p.y - monster.y))
                
                # Simple A* pathfinding (just one step)
                best_move = (monster.x, monster.y)
                min_dist = abs(nearest_player.x - monster.x) + abs(nearest_player.y - monster.y)

                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    nx, ny = monster.x + dx, monster.y + dy
                    if (nx, ny) in all_units_coords and (nx, ny) != (monster.x, monster.y):
                        continue
                    if 0 <= nx < self.GRID_SIZE[0] and 0 <= ny < self.GRID_SIZE[1]:
                        dist = abs(nearest_player.x - nx) + abs(nearest_player.y - ny)
                        if dist < min_dist:
                            min_dist = dist
                            best_move = (nx, ny)
                
                if best_move != (monster.x, monster.y):
                    monster.x, monster.y = best_move
        
        return reward

    def _cleanup_dead_units(self) -> float:
        reward = 0.0
        
        killed_monsters = sum(1 for m in self.monster_units if not m.is_alive)
        reward += 1.0 * killed_monsters
        if killed_monsters > 0:
            self.score += killed_monsters

        self.player_units = [u for u in self.player_units if u.is_alive]
        self.monster_units = [m for m in self.monster_units if m.is_alive]

        if self.selected_unit_idx >= len(self.player_units) and self.player_units:
            self.selected_unit_idx = 0

        return reward

    def _cycle_selection(self, forward: bool, force_first: bool = False):
        if not self.player_units:
            return
        
        if force_first:
            self.selected_unit_idx = 0
            return
            
        num_units = len(self.player_units)
        direction = 1 if forward else -1
        self.selected_unit_idx = (self.selected_unit_idx + direction) % num_units

    def _is_valid_move(self, x: int, y: int) -> bool:
        if not (0 <= x < self.GRID_SIZE[0] and 0 <= y < self.GRID_SIZE[1]):
            return False
        
        occupied_coords = {(u.x, u.y) for u in self.player_units + self.monster_units}
        if (x, y) in occupied_coords:
            return False
            
        return True

    def _find_adjacent_target(self, attacker: Unit, targets: List[Unit]) -> Optional[Unit]:
        for target in targets:
            if target.is_alive and abs(attacker.x - target.x) + abs(attacker.y - target.y) == 1:
                return target
        return None

    def _get_observation(self) -> np.ndarray:
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self) -> dict:
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "player_units": len(self.player_units),
            "monster_units": len(self.monster_units),
        }

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_SIZE[0] + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x * self.CELL_SIZE, 0), (x * self.CELL_SIZE, self.SCREEN_HEIGHT))
        for y in range(self.GRID_SIZE[1] + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y * self.CELL_SIZE), (self.SCREEN_WIDTH, y * self.CELL_SIZE))

        # Draw selection and range indicators
        if self.player_units and not self.player_units[self.selected_unit_idx].has_acted:
            selected_unit = self.player_units[self.selected_unit_idx]
            
            # Pulsing selection circle
            pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2
            radius = int(self.CELL_SIZE * 0.4 * (1 + pulse * 0.1))
            center = (int(selected_unit.x * self.CELL_SIZE + self.CELL_SIZE/2), int(selected_unit.y * self.CELL_SIZE + self.CELL_SIZE/2))
            pygame.gfxdraw.aacircle(self.screen, center[0], center[1], radius, self.COLOR_SELECT)

            # Draw move/attack range
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = selected_unit.x + dx, selected_unit.y + dy
                rect = (nx * self.CELL_SIZE, ny * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                
                is_monster = any(m.x == nx and m.y == ny for m in self.monster_units)
                is_valid_move = self._is_valid_move(nx, ny)

                if is_monster:
                    s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                    s.fill(self.COLOR_ATTACK)
                    self.screen.blit(s, (rect[0], rect[1]))
                elif is_valid_move:
                    s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                    s.fill(self.COLOR_MOVE)
                    self.screen.blit(s, (rect[0], rect[1]))

        # Draw units
        all_units = self.player_units + self.monster_units
        for unit in sorted(all_units, key=lambda u: u.y): # Draw from top to bottom
            unit.draw(self.screen, self.CELL_SIZE)

        # Draw health bars on top of units
        for unit in all_units:
            unit.draw_health_bar(self.screen, self.CELL_SIZE)
        
        # Draw and update animations
        self.animations = [anim for anim in self.animations if anim.update()]
        for anim in self.animations:
            anim.draw(self.screen)

    def _render_ui(self):
        score_text = self.font_m.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 5))
        
        wave_text = self.font_m.render(f"WAVE: {self.wave}", True, self.COLOR_UI)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 5))

        if self.game_over:
            end_text = self.font_m.render("GAME OVER", True, (255, 50, 50))
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # Requires `pip install pygame`
    
    # Unset the dummy driver to allow display
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Tactical Grid Combat")
    clock = pygame.time.Clock()
    
    done = False
    
    print("\n" + "="*30)
    print("MANUAL PLAY MODE")
    print(env.user_guide)
    print("="*30 + "\n")

    while not done:
        # This is event-driven, so we check for key presses once per frame
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                movement = 0 # 0=none
                space_held = 0 # 0=released
                shift_held = 0 # 0=released

                if event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                elif event.key == pygame.K_SPACE:
                    space_held = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_held = 1
                elif event.key == pygame.K_RETURN: # Use Enter for no-op/end turn
                    movement, space_held, shift_held = 0, 0, 0
                
                # Since auto_advance is False, we call step() on each key press
                action = [movement, space_held, shift_held]
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}, Done: {done}")
                action_taken = True
        
        if done:
            break

        # Draw the observation from the environment
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)

    env.close()
    print("Game Over!")