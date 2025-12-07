import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import random
import os
import pygame


# Set Pygame to run in a headless mode
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

class GameEnv(gym.Env):
    """
    A turn-based tactical game where you command a squad of units against waves of advancing centipedes.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    # --- Game Metadata ---
    game_description = (
        "A turn-based tactical game where you command a squad of units against waves of advancing centipedes."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your active unit. Press space to attack the nearest enemy, "
        "and press shift to hunker down and reduce incoming damage."
    )
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.CELL_SIZE = self.HEIGHT // self.GRID_SIZE
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.UI_WIDTH = self.WIDTH - self.GRID_WIDTH
        self.MAX_STEPS = 1000

        # --- Player Unit Constants ---
        self.NUM_UNITS = 3
        self.UNIT_RANGE = 3
        self.UNIT_AMMO_START = 10
        self.UNIT_HEALTH_START = 1

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_INACTIVE = (0, 160, 80)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_HIGHLIGHT = (255, 255, 0)
        self.COLOR_RANGE = (0, 100, 255, 70) # RGBA for transparency
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_PROJECTILE = (255, 255, 255)
        self.COLOR_EXPLOSION = (255, 150, 0)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_title = pygame.font.SysFont("Consolas", 32, bold=True)
        self.range_surface = pygame.Surface((self.GRID_WIDTH, self.HEIGHT), pygame.SRCALPHA)

        # --- Game State Initialization ---
        self.steps = 0
        self.score = 0
        self.wave_number = 0
        self.game_over = False
        self.player_units = []
        self.centipedes = []
        self.active_unit_idx = 0
        self.visual_effects = [] # For projectiles and explosions

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.wave_number = 1
        self.game_over = False
        self.active_unit_idx = 0
        self.visual_effects = []

        # --- Initialize Player Units ---
        self.player_units = []
        for i in range(self.NUM_UNITS):
            self.player_units.append({
                "pos": (i * 2 + 2, self.GRID_SIZE - 1),
                "health": self.UNIT_HEALTH_START,
                "ammo": self.UNIT_AMMO_START,
                "hunkered": False,
                "is_dead": False
            })

        # --- Initialize First Centipede Wave ---
        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        self.visual_effects = []
        
        # Reset hunkered status for all units at the start of a player's turn cycle
        if self.active_unit_idx == 0:
            for unit in self.player_units:
                unit["hunkered"] = False

        # --- Player Turn for Active Unit ---
        reward += self._handle_player_action(action)

        # --- Advance Turn ---
        self.active_unit_idx = (self.active_unit_idx + 1)
        if self.active_unit_idx >= len(self.player_units):
            self.active_unit_idx = 0
            # --- Enemy Turn (executes after all player units have acted) ---
            reward += self._handle_enemy_turn()

        # --- Post-Turn Cleanup & State Update ---
        # Remove dead units
        living_units = [u for u in self.player_units if not u["is_dead"]]
        if len(living_units) < len(self.player_units):
            reward -= 1.0 * (len(self.player_units) - len(living_units))
            self.player_units = living_units
            if self.active_unit_idx >= len(self.player_units):
                self.active_unit_idx = 0
        
        # Remove empty centipede chains
        self.centipedes = [c for c in self.centipedes if c]

        # Check for wave clear
        if not self.centipedes and not self.game_over:
            reward += 10.0 # Wave clear bonus
            self.wave_number += 1
            self._spawn_wave()
            # Refill some ammo for the next wave
            for unit in self.player_units:
                unit["ammo"] = min(self.UNIT_AMMO_START, unit["ammo"] + 5)


        # --- Check Termination Conditions ---
        terminated = False
        if not self.player_units:
            reward -= 100.0
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        self.score += reward

        truncated = False # This environment does not truncate based on time limits.

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_player_action(self, action):
        reward = 0
        if not self.player_units or self.active_unit_idx >= len(self.player_units): return 0

        movement, space_held, shift_held = action
        active_unit = self.player_units[self.active_unit_idx]

        if active_unit["is_dead"]:
             return 0

        # Action Priority: Attack > Hunker > Move > Wait
        if space_held == 1 and active_unit["ammo"] > 0:
            # ATTACK
            target, dist = self._find_closest_enemy(active_unit["pos"])
            if target and dist <= self.UNIT_RANGE:
                active_unit["ammo"] -= 1
                # Add projectile visual effect
                self.visual_effects.append({
                    "type": "projectile",
                    "start": self._grid_to_pixel(*active_unit["pos"]),
                    "end": self._grid_to_pixel(*target["pos"]),
                })
                reward += self._damage_centipede(target)
        elif shift_held == 1:
            # HUNKER DOWN
            active_unit["hunkered"] = True
        elif movement != 0:
            # MOVE
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            new_pos = (active_unit["pos"][0] + dx, active_unit["pos"][1] + dy)
            if self._is_valid_move(new_pos):
                active_unit["pos"] = new_pos
        
        return reward

    def _handle_enemy_turn(self):
        reward = 0
        all_segments = [seg for chain in self.centipedes for seg in chain]
        
        for segment in all_segments:
            if not self.player_units: break
            
            # Find closest player unit
            closest_unit = min(self.player_units, key=lambda u: self._distance(segment["pos"], u["pos"]))
            
            # Move towards the unit
            dx = np.sign(closest_unit["pos"][0] - segment["pos"][0])
            dy = np.sign(closest_unit["pos"][1] - segment["pos"][1])
            new_pos = (segment["pos"][0] + dx, segment["pos"][1] + dy)
            segment["pos"] = new_pos

            # Check for collision
            for i, unit in enumerate(self.player_units):
                if unit["pos"] == segment["pos"] and not unit["is_dead"]:
                    if not unit["hunkered"]:
                        unit["health"] -= 1
                        reward -= 0.1
                        if unit["health"] <= 0:
                            unit["is_dead"] = True
                    
                    self.visual_effects.append({
                        "type": "explosion", "pos": self._grid_to_pixel(*unit["pos"]), "radius": self.CELL_SIZE * 0.8
                    })
                    segment["is_dead"] = True # Mark segment for removal
                    break # A segment can only damage one unit

        # Remove dead segments and split chains
        new_centipedes = []
        for chain in self.centipedes:
            new_chain = []
            for segment in chain:
                if not segment.get("is_dead", False):
                    new_chain.append(segment)
                else:
                    if new_chain:
                        new_centipedes.append(new_chain)
                    new_chain = []
            if new_chain:
                new_centipedes.append(new_chain)
        self.centipedes = new_centipedes

        return reward

    def _damage_centipede(self, target_segment):
        reward = 0.1
        target_segment["health"] -= 1
        self.visual_effects.append({
            "type": "explosion", "pos": self._grid_to_pixel(*target_segment["pos"]), "radius": self.CELL_SIZE * 0.6
        })

        if target_segment["health"] <= 0:
            reward += 1.0
            
            # Find and split the centipede chain
            for i, chain in enumerate(self.centipedes):
                if target_segment in chain:
                    try:
                        seg_idx = chain.index(target_segment)
                        # Split into two new chains if not a head or tail
                        chain_before = chain[:seg_idx]
                        chain_after = chain[seg_idx+1:]
                        
                        del self.centipedes[i]
                        if chain_before: self.centipedes.insert(i, chain_before)
                        if chain_after: self.centipedes.insert(i, chain_after)
                        
                    except ValueError:
                        pass # Segment already removed
                    break
        return reward

    def _spawn_wave(self):
        centipede_len = 4 + self.wave_number
        start_x = self.GRID_SIZE // 2
        new_chain = []
        for i in range(centipede_len):
            new_chain.append({
                "pos": (start_x, i),
                "health": 1
            })
        self.centipedes.append(new_chain)

    def _get_observation(self):
        # --- Main Drawing Surface ---
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_range_indicator()
        self._render_centipedes()
        self._render_units()
        self._render_effects()
        self._render_ui()

        # Convert to numpy array (H, W, C format)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for x in range(self.GRID_SIZE + 1):
            px = x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.HEIGHT), 1)
        for y in range(self.GRID_SIZE + 1):
            py = y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.GRID_WIDTH, py), 1)

    def _render_range_indicator(self):
        self.range_surface.fill((0,0,0,0))
        if self.player_units and not self.game_over and self.active_unit_idx < len(self.player_units):
            active_unit = self.player_units[self.active_unit_idx]
            if not active_unit["is_dead"]:
                ax, ay = active_unit["pos"]
                for x in range(self.GRID_SIZE):
                    for y in range(self.GRID_SIZE):
                        if self._distance((ax, ay), (x, y)) <= self.UNIT_RANGE:
                            px, py = self._grid_to_pixel(x, y, center=False)
                            pygame.draw.rect(self.range_surface, self.COLOR_RANGE, (px, py, self.CELL_SIZE, self.CELL_SIZE))
        self.screen.blit(self.range_surface, (0, 0))

    def _render_centipedes(self):
        radius = int(self.CELL_SIZE * 0.4)
        for chain in self.centipedes:
            for i, segment in enumerate(chain):
                px, py = self._grid_to_pixel(*segment["pos"])
                pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_ENEMY)
                pygame.gfxdraw.filled_circle(self.screen, px, py, radius, self.COLOR_ENEMY)
                # Draw lines connecting segments
                if i > 0:
                    prev_px, prev_py = self._grid_to_pixel(*chain[i-1]["pos"])
                    pygame.draw.aaline(self.screen, self.COLOR_ENEMY, (px, py), (prev_px, prev_py))


    def _render_units(self):
        radius = int(self.CELL_SIZE * 0.4)
        for i, unit in enumerate(self.player_units):
            if unit["is_dead"]: continue
            
            px, py = self._grid_to_pixel(*unit["pos"])
            color = self.COLOR_PLAYER if i == self.active_unit_idx else self.COLOR_PLAYER_INACTIVE

            # Highlight for active unit
            if i == self.active_unit_idx:
                pygame.gfxdraw.aacircle(self.screen, px, py, int(radius * 1.5), self.COLOR_HIGHLIGHT)
                pygame.gfxdraw.filled_circle(self.screen, px, py, int(radius * 1.5), self.COLOR_HIGHLIGHT)

            # Hunker down effect
            if unit["hunkered"]:
                 pygame.gfxdraw.aacircle(self.screen, px, py, int(radius * 1.2), (100, 100, 255))
                 pygame.gfxdraw.filled_circle(self.screen, px, py, int(radius * 1.2), (100, 100, 255))

            # Main unit circle
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, color)
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, color)

            # Health/Ammo text
            ammo_text = self.font_small.render(f"A:{unit['ammo']}", True, self.COLOR_TEXT)
            self.screen.blit(ammo_text, (px - ammo_text.get_width() // 2, py + radius - 5))

    def _render_effects(self):
        for effect in self.visual_effects:
            if effect["type"] == "projectile":
                pygame.draw.aaline(self.screen, self.COLOR_PROJECTILE, effect["start"], effect["end"], blend=True)
            elif effect["type"] == "explosion":
                pygame.gfxdraw.aacircle(self.screen, effect["pos"][0], effect["pos"][1], int(effect["radius"]), self.COLOR_EXPLOSION)
                pygame.gfxdraw.filled_circle(self.screen, effect["pos"][0], effect["pos"][1], int(effect["radius"]), self.COLOR_EXPLOSION)

    def _render_ui(self):
        ui_x = self.GRID_WIDTH + 20
        # Title
        title_surf = self.font_title.render("CENTIPEDE TACTICS", True, self.COLOR_HIGHLIGHT)
        self.screen.blit(title_surf, (ui_x, 20))
        
        # Stats
        score_surf = self.font_large.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (ui_x, 80))
        
        wave_surf = self.font_large.render(f"Wave: {self.wave_number}", True, self.COLOR_TEXT)
        self.screen.blit(wave_surf, (ui_x, 110))

        steps_surf = self.font_large.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_surf, (ui_x, 140))

        # Active Unit Info
        if self.player_units and not self.game_over and self.active_unit_idx < len(self.player_units):
            active_unit = self.player_units[self.active_unit_idx]
            info_y = 200
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_WIDTH+10, info_y-10), (self.WIDTH-10, info_y-10), 2)
            
            active_title = self.font_large.render(f"Unit {self.active_unit_idx + 1} Turn", True, self.COLOR_HIGHLIGHT)
            self.screen.blit(active_title, (ui_x, info_y))
            
            ammo_surf = self.font_large.render(f"Ammo: {active_unit['ammo']}", True, self.COLOR_TEXT)
            self.screen.blit(ammo_surf, (ui_x, info_y + 40))

            health_surf = self.font_large.render(f"Health: {active_unit['health']}", True, self.COLOR_TEXT)
            self.screen.blit(health_surf, (ui_x, info_y + 70))
        
        if self.game_over:
            reason = "ALL UNITS LOST" if not self.player_units else "TIME LIMIT REACHED"
            end_surf = self.font_title.render("GAME OVER", True, self.COLOR_ENEMY)
            reason_surf = self.font_large.render(reason, True, self.COLOR_ENEMY)
            self.screen.blit(end_surf, (self.WIDTH // 2 - end_surf.get_width() // 2, self.HEIGHT // 2 - 40))
            self.screen.blit(reason_surf, (self.WIDTH // 2 - reason_surf.get_width() // 2, self.HEIGHT // 2))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "active_unit": self.active_unit_idx,
            "player_units_left": len(self.player_units),
            "centipede_segments": sum(len(c) for c in self.centipedes)
        }

    # --- Helper Functions ---
    def _grid_to_pixel(self, x, y, center=True):
        px = x * self.CELL_SIZE
        py = y * self.CELL_SIZE
        if center:
            px += self.CELL_SIZE // 2
            py += self.CELL_SIZE // 2
        return int(px), int(py)

    def _distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) # Manhattan distance

    def _is_valid_move(self, pos):
        x, y = pos
        if not (0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE):
            return False
        for unit in self.player_units:
            if unit["pos"] == pos and not unit["is_dead"]:
                return False
        return True

    def _find_closest_enemy(self, pos):
        all_segments = [seg for chain in self.centipedes for seg in chain]
        if not all_segments:
            return None, float('inf')
        
        closest_seg = min(all_segments, key=lambda s: self._distance(pos, s["pos"]))
        dist = self._distance(pos, closest_seg["pos"])
        return closest_seg, dist

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Manual Play Loop ---
    # To run this, you'll need to unset the dummy video driver
    # and install pygame with display support.
    # For example:
    # del os.environ['SDL_VIDEODRIVER']
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Un-comment the next line to run with a display
    # screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    # pygame.display.set_caption("Centipede Tactics")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    # This loop is for demonstration and will not run in headless mode
    # without modifications. It shows how keyboard inputs map to actions.
    print("Running in headless mode. Manual play loop is for demonstration purposes.")
    
    # Example of running a few steps with random actions
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
        if terminated or truncated:
            print("Game ended.")
            env.reset()

    env.close()
    print("Demonstration finished.")