import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:58:06.331918
# Source Brief: brief_00188.md
# Brief Index: 188
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "In a neon-drenched arena, control a quantum entity that can create clones of itself. "
        "Strategically place clones to ambush and destroy patrolling guards while managing your energy."
    )
    user_guide = (
        "Controls: Use arrow keys to move. Press space to create a clone. "
        "Press shift to teleport to the nearest clone."
    )
    auto_advance = False

    # --- CONSTANTS ---
    SCREEN_W, SCREEN_H = 640, 400
    GRID_W, GRID_H = 32, 20
    CELL_SIZE = 20
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_GRID = (30, 40, 60)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_PLAYER_GLOW = (0, 255, 150, 50)
    COLOR_PLAYER_ACTIVE = (255, 255, 255)
    COLOR_GUARD = (255, 50, 50)
    COLOR_GUARD_GLOW = (255, 50, 50, 60)
    COLOR_GUARD_PATH = (80, 40, 40)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_ENERGY_BAR = (0, 200, 100)
    COLOR_ENERGY_BAR_BG = (40, 40, 40)
    COLOR_TELEPORT_LINE = (255, 255, 255)

    # Game Parameters
    INITIAL_GUARDS = 2
    MAX_GUARDS = 6
    MAX_ENERGY = 100.0
    CLONE_COST = 25.0
    ENERGY_REGEN_PER_STEP = 0.25
    GUARD_BASE_MOVE_COOLDOWN = 20 # Steps per move at start
    GUARD_SPEED_INCREASE_FACTOR = 0.15 # Speeds up per defeated guard
    PARTICLE_LIFESPAN = 20

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_W, self.SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.clones = []
        self.active_clone_idx = 0
        self.guards = []
        self.energy = 0.0
        self.defeated_guards_count = 0
        self.particles = []
        self.teleport_effect = None
        self.prev_space_held = False
        self.prev_shift_held = False
        
        # self.reset() is called by the environment wrapper

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.defeated_guards_count = 0
        self.energy = self.MAX_ENERGY
        self.particles = []
        self.teleport_effect = None
        self.prev_space_held = False
        self.prev_shift_held = False

        # Initialize player
        start_pos = (self.GRID_W // 4, self.GRID_H // 2)
        self.clones = [{'pos': start_pos}]
        self.active_clone_idx = 0

        # Initialize guards
        self.guards = []
        num_guards = self.INITIAL_GUARDS
        occupied_starts = {start_pos}
        for _ in range(num_guards):
            path = self._generate_guard_path(occupied_starts)
            if path:
                occupied_starts.add(path[0])
                self.guards.append({
                    'pos': path[0],
                    'path': path,
                    'waypoint_idx': 1,
                    'move_timer': self.np_random.integers(0, self.GUARD_BASE_MOVE_COOLDOWN)
                })
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Player Input ---
        self._handle_input(movement, space_held, shift_held)
        
        # --- Update Game Logic ---
        self._update_guards()
        self.energy = min(self.MAX_ENERGY, self.energy + self.ENERGY_REGEN_PER_STEP)
        self._update_particles()
        if self.teleport_effect and self.teleport_effect['timer'] > 0:
            self.teleport_effect['timer'] -= 1

        # --- Resolve Conflicts ---
        conflict_reward = self._resolve_conflicts()
        reward += conflict_reward
        
        # --- Update Game State ---
        self._update_active_clone_after_destruction()
        
        # --- Calculate Rewards ---
        proximity_reward = self._calculate_proximity_reward()
        reward += proximity_reward
        reward -= 0.01 # Small time penalty to encourage efficiency

        self.score += reward
        
        # --- Check Termination ---
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        self.score += terminal_reward
        self.game_over = terminated
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        if not self.clones: return

        active_clone = self.clones[self.active_clone_idx]
        
        # --- Action: Teleport (Shift, rising edge) ---
        teleported = False
        if shift_held and not self.prev_shift_held and len(self.clones) > 1:
            # sound: teleport_activate.wav
            self._teleport_to_nearest_clone()
            teleported = True

        # --- Action: Clone (Space, rising edge) ---
        # Teleport takes precedence over cloning if both are pressed on the same frame
        if not teleported and space_held and not self.prev_space_held:
            if self.energy >= self.CLONE_COST:
                # sound: clone_create.wav
                self.energy -= self.CLONE_COST
                self.clones.append({'pos': active_clone['pos']})
                self._spawn_particles(active_clone['pos'], self.COLOR_PLAYER, 10)

        # --- Action: Movement ---
        if movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            new_x = max(0, min(self.GRID_W - 1, active_clone['pos'][0] + dx))
            new_y = max(0, min(self.GRID_H - 1, active_clone['pos'][1] + dy))
            active_clone['pos'] = (new_x, new_y)

    def _teleport_to_nearest_clone(self):
        active_pos = self.clones[self.active_clone_idx]['pos']
        min_dist = float('inf')
        nearest_idx = -1

        for i, clone in enumerate(self.clones):
            if i == self.active_clone_idx: continue
            dist = self._manhattan_distance(active_pos, clone['pos'])
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        if nearest_idx != -1:
            old_pos_px = self._grid_to_pixel(self.clones[self.active_clone_idx]['pos'])
            new_pos_px = self._grid_to_pixel(self.clones[nearest_idx]['pos'])
            self.active_clone_idx = nearest_idx
            self.teleport_effect = {'from': old_pos_px, 'to': new_pos_px, 'timer': 10}

    def _update_guards(self):
        cooldown = self.GUARD_BASE_MOVE_COOLDOWN / (1 + self.GUARD_SPEED_INCREASE_FACTOR * self.defeated_guards_count)
        for guard in self.guards:
            guard['move_timer'] += 1
            if guard['move_timer'] >= cooldown:
                # sound: guard_step.wav
                guard['move_timer'] = 0
                target_pos = guard['path'][guard['waypoint_idx']]
                
                # Move towards target
                dx = target_pos[0] - guard['pos'][0]
                dy = target_pos[1] - guard['pos'][1]
                
                if abs(dx) > abs(dy):
                    guard['pos'] = (guard['pos'][0] + np.sign(dx), guard['pos'][1])
                elif abs(dy) > 0:
                    guard['pos'] = (guard['pos'][0], guard['pos'][1] + np.sign(dy))

                if guard['pos'] == target_pos:
                    guard['waypoint_idx'] = (guard['waypoint_idx'] + 1) % len(guard['path'])

    def _resolve_conflicts(self):
        reward = 0
        clone_pos_set = {c['pos'] for c in self.clones}
        guard_pos_map = {g['pos']: [] for g in self.guards}
        for i, g in enumerate(self.guards):
            guard_pos_map[g['pos']].append(i)

        conflict_cells = clone_pos_set.intersection(guard_pos_map.keys())
        
        if not conflict_cells:
            return 0

        clones_to_remove = set()
        guards_to_remove = set()

        for cell in conflict_cells:
            # Mark all clones and guards on this cell for removal
            for i, clone in enumerate(self.clones):
                if clone['pos'] == cell:
                    clones_to_remove.add(i)
            
            for guard_idx in guard_pos_map.get(cell, []):
                guards_to_remove.add(guard_idx)

        # Remove guards and give rewards
        if guards_to_remove:
            # sound: guard_destroy.wav
            for idx in sorted(list(guards_to_remove), reverse=True):
                self._spawn_particles(self.guards[idx]['pos'], self.COLOR_GUARD, 20)
                del self.guards[idx]
                reward += 1.0
                self.defeated_guards_count += 1
        
        # Remove clones
        if clones_to_remove:
            # sound: clone_destroy.wav
            for idx in sorted(list(clones_to_remove), reverse=True):
                if self.clones: # Check if list is not already empty
                    self._spawn_particles(self.clones[idx]['pos'], self.COLOR_PLAYER, 15)
                    del self.clones[idx]

        return reward

    def _update_active_clone_after_destruction(self):
        if not self.clones:
            self.active_clone_idx = -1
        elif self.active_clone_idx >= len(self.clones):
            self.active_clone_idx = 0 # Default to the first available clone

    def _calculate_proximity_reward(self):
        reward = 0
        if not self.clones or not self.guards:
            return 0
        
        for clone in self.clones:
            for guard in self.guards:
                dist = self._manhattan_distance(clone['pos'], guard['pos'])
                if dist <= 3:
                    reward += 0.01 # Reduced from 0.1 to keep rewards small
        return reward

    def _check_termination(self):
        if not self.clones:
            return True, -100.0 # Lost
        if not self.guards:
            return True, 100.0 # Won
        # Time limit is handled by truncation
        return False, 0.0

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
            "energy": self.energy,
            "clones": len(self.clones),
            "guards": len(self.guards),
        }

    # --- RENDERING METHODS ---
    def _render_game(self):
        self._draw_grid()
        self._draw_guard_paths()
        self._draw_teleport_effect()
        self._draw_guards()
        self._draw_clones()
        self._draw_particles()

    def _render_ui(self):
        # Energy Bar
        bar_w, bar_h = 150, 20
        pygame.draw.rect(self.screen, self.COLOR_ENERGY_BAR_BG, (10, 10, bar_w, bar_h))
        energy_w = max(0, int((self.energy / self.MAX_ENERGY) * bar_w))
        pygame.draw.rect(self.screen, self.COLOR_ENERGY_BAR, (10, 10, energy_w, bar_h))
        
        # Guards Remaining
        guards_text = self.font_large.render(f"GUARDS: {len(self.guards)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(guards_text, (self.SCREEN_W - guards_text.get_width() - 10, 10))

        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_W // 2 - score_text.get_width() // 2, 10))

    def _draw_grid(self):
        for x in range(0, self.SCREEN_W, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_H))
        for y in range(0, self.SCREEN_H, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_W, y))

    def _draw_guard_paths(self):
        for guard in self.guards:
            if len(guard['path']) > 1:
                path_pixels = [self._grid_to_pixel(p) for p in guard['path']]
                pygame.draw.lines(self.screen, self.COLOR_GUARD_PATH, True, path_pixels, 2)

    def _draw_guards(self):
        for guard in self.guards:
            px, py = self._grid_to_pixel(guard['pos'])
            radius = self.CELL_SIZE // 2 - 2
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius + 3, self.COLOR_GUARD_GLOW)
            # Triangle body
            p1 = (px, py - radius)
            p2 = (px - radius, py + radius // 2)
            p3 = (px + radius, py + radius // 2)
            pygame.gfxdraw.filled_trigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], self.COLOR_GUARD)
            pygame.gfxdraw.aatrigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], self.COLOR_GUARD)

    def _draw_clones(self):
        radius = self.CELL_SIZE // 2 - 3
        for i, clone in enumerate(self.clones):
            px, py = self._grid_to_pixel(clone['pos'])
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius + 4, self.COLOR_PLAYER_GLOW)
            # Circle body
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_PLAYER)
            # Active clone indicator
            if i == self.active_clone_idx:
                pygame.gfxdraw.aacircle(self.screen, px, py, radius + 2, self.COLOR_PLAYER_ACTIVE)
                pygame.gfxdraw.aacircle(self.screen, px, py, radius + 3, self.COLOR_PLAYER_ACTIVE)

    def _draw_teleport_effect(self):
        if self.teleport_effect and self.teleport_effect['timer'] > 0:
            alpha = int(255 * (self.teleport_effect['timer'] / 10))
            color = (*self.COLOR_TELEPORT_LINE, alpha)
            p1 = self.teleport_effect['from']
            p2 = self.teleport_effect['to']
            # Draw multiple lines for thickness with alpha
            for i in range(-1, 2):
                pygame.draw.aaline(self.screen, color, (p1[0]+i, p1[1]), (p2[0]+i, p2[1]), 1)
                pygame.draw.aaline(self.screen, color, (p1[0], p1[1]+i), (p2[0], p2[1]+i), 1)

    def _draw_particles(self):
        for p in self.particles:
            life_ratio = p['life'] / self.PARTICLE_LIFESPAN
            px, py = self._grid_to_pixel(p['pos'])
            px += int(p['vel'][0] * (self.PARTICLE_LIFESPAN - p['life']))
            py += int(p['vel'][1] * (self.PARTICLE_LIFESPAN - p['life']))
            radius = max(0, int((self.CELL_SIZE / 4) * life_ratio))
            color = (*p['color'], int(255 * life_ratio))
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, color)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['life'] -= 1

    # --- UTILITY METHODS ---
    def _grid_to_pixel(self, grid_pos):
        px = int(grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2)
        py = int(grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2)
        return px, py

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _generate_guard_path(self, occupied_starts):
        for _ in range(20): # Try 20 times to find a valid path
            path_len = self.np_random.integers(2, 5)
            path = []
            last_pos = None
            for i in range(path_len):
                for _ in range(10): # Try 10 times to find a valid point
                    px = self.np_random.integers(0, self.GRID_W)
                    py = self.np_random.integers(0, self.GRID_H)
                    pos = (px, py)
                    if i == 0 and pos in occupied_starts: continue
                    if last_pos and self._manhattan_distance(pos, last_pos) < 5: continue
                    path.append(pos)
                    last_pos = pos
                    break
            if len(path) == path_len:
                return path
        return []

    def _spawn_particles(self, grid_pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 2.0)
            self.particles.append({
                'pos': grid_pos,
                'vel': (math.cos(angle) * speed, math.sin(angle) * speed),
                'life': self.PARTICLE_LIFESPAN,
                'color': color
            })

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It is NOT part of the required environment implementation
    
    # Un-comment the next line to run with display
    # os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # For manual play, we need a real display.
    if os.environ.get("SDL_VIDEODRIVER") != "dummy":
        screen = pygame.display.set_mode((GameEnv.SCREEN_W, GameEnv.SCREEN_H))
        pygame.display.set_caption("Quantum Arena")
        clock = pygame.time.Clock()

        running = True
        while running:
            movement = 0 # No-op
            space_held = 0
            shift_held = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            if keys[pygame.K_SPACE]: space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Draw the observation to the display screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
                # Wait for 'R' to reset
                wait_for_reset = True
                while wait_for_reset:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            wait_for_reset = False
                            running = False
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                            obs, info = env.reset()
                            wait_for_reset = False
                    clock.tick(30)
            
            clock.tick(30) # Run at 30 FPS

        env.close()
    else:
        print("Running in headless mode. No display will be shown.")
        print("To play manually, set a valid SDL_VIDEODRIVER.")
        # You can still run a basic test loop
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"Episode finished. Score: {info['score']}")
                env.reset()
        env.close()
        print("Headless test complete.")