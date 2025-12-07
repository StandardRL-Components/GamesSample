import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:43:12.617466
# Source Brief: brief_01865.md
# Brief Index: 1865
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A rhythm-stealth game where the player, a bioluminescent cephalopod,
    teleports between colored tiles to camouflage and infiltrate an underwater facility.
    The agent must match tile patterns on the beat to remain hidden from guards.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "In this rhythm-stealth game, teleport your bioluminescent cephalopod between colored tiles on the beat to stay camouflaged and avoid guards."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to select a tile to teleport to. Press space on the beat to execute the teleport."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 16, 10
    TILE_SIZE = 40
    FINAL_ZONE = 5
    MAX_STEPS = 2500

    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_GUARD = (255, 50, 50)
    COLOR_GUARD_CONE = (255, 100, 100, 70)
    COLOR_TARGET_VALID = (255, 255, 0)
    COLOR_TARGET_SELECTED = (0, 255, 100)
    COLOR_TEXT = (220, 220, 240)
    PATTERN_COLORS = [
        (40, 60, 90),    # 0: Rock (default)
        (20, 120, 120),  # 1: Coral
        (160, 140, 80),  # 2: Sand
        (100, 50, 150),  # 3: Anemone
        (80, 180, 50),   # 4: Seaweed
    ]
    CHAIN_REACTION_COLOR = (255, 150, 0)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # --- Game State Initialization ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_zone = 0
        self.beat_duration = 30  # 30 steps per beat (1 sec @ 30fps)
        self.beat_timer = 0
        
        self.player_grid_pos = (0, 0)
        self.player_render_pos = (0, 0)
        self.player_last_grid_pos = (0, 0)
        self.player_pattern = 1
        self.unlocked_patterns = {1}
        self.is_camouflaged = False
        
        self.grid = []
        self.guards = []
        self.particles = deque()
        self.distractions = []

        self.valid_targets = []
        self.selected_target_idx = -1
        
        self.detection_level = 0.0
        self.new_zone_transition = 0.0

        # self.reset() is called by the wrapper/runner
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_zone = 1
        
        self.beat_timer = 0
        self.detection_level = 0.0
        self.new_zone_transition = 0.0

        self.unlocked_patterns = {1}
        self.player_pattern = 1
        
        self.guards = []
        self.particles.clear()
        self.distractions = []
        
        self._generate_zone()
        self.player_render_pos = self._grid_to_pixel(self.player_grid_pos)
        self.player_last_grid_pos = self.player_grid_pos
        self.is_camouflaged = True # Start camouflaged
        
        self._update_valid_targets()
        self.selected_target_idx = -1
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.beat_timer = (self.beat_timer + 1) % self.beat_duration
        is_on_beat = self.beat_timer == 0

        # --- Action Processing ---
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement)
        
        if is_on_beat:
            # --- Beat-based Game Logic ---
            reward += self._update_beat_logic(space_pressed, shift_pressed)
        else:
            # --- Continuous Game Logic ---
            reward += self._update_continuous_logic()

        self._update_effects()
        self.steps += 1
        
        # --- Termination Check ---
        truncated = self.steps >= self.MAX_STEPS
        terminated = self.game_over
        if terminated and not truncated: # Win condition
             reward += 100
        
        self.score += reward
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement_action):
        """Updates the selected target based on directional input."""
        if not self.valid_targets or movement_action == 0:
            self.selected_target_idx = -1
            return

        player_x, player_y = self.player_grid_pos
        best_target_idx = -1
        min_dist_sq = float('inf')

        # Define direction vectors
        dirs = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # Up, Down, Left, Right
        target_dir = dirs[movement_action]

        for i, target_pos in enumerate(self.valid_targets):
            vec_to_target = (target_pos[0] - player_x, target_pos[1] - player_y)
            
            # Prioritize targets in the chosen direction
            dot_product = vec_to_target[0] * target_dir[0] + vec_to_target[1] * target_dir[1]
            if dot_product > 0:
                dist_sq = vec_to_target[0]**2 + vec_to_target[1]**2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_target_idx = i
        
        self.selected_target_idx = best_target_idx


    def _update_beat_logic(self, space_pressed, shift_pressed):
        """Handles all game logic that occurs on the beat."""
        reward = 0
        
        # Player Teleport
        teleported = False
        if space_pressed and self.selected_target_idx != -1:
            target_pos = self.valid_targets[self.selected_target_idx]
            
            self.player_last_grid_pos = self.player_grid_pos
            self.player_grid_pos = target_pos
            
            # sfx: player_teleport.wav
            self._create_particles(self._grid_to_pixel(self.player_last_grid_pos), self.COLOR_PLAYER, 20)
            self._create_particles(self._grid_to_pixel(self.player_grid_pos), self.COLOR_PLAYER, 20)
            
            tile = self.grid[self.player_grid_pos[1]][self.player_grid_pos[0]]
            self.player_pattern = tile['pattern']
            self.is_camouflaged = self.player_pattern in self.unlocked_patterns
            teleported = True

            if self.is_camouflaged:
                reward += 1.0 # Successful camouflage
                # sfx: camouflage_success.wav
            
            if tile['is_chain_reaction']:
                reward += 5.0
                self._trigger_chain_reaction(self.player_grid_pos)
                tile['is_chain_reaction'] = False # one-time use
                # sfx: chain_reaction.wav
        
        # Update guards and detection
        self._update_guards()
        detection_penalty = self._update_detection()
        reward += detection_penalty

        # Check for reaching exit
        if self.player_grid_pos == self.exit_pos:
            reward += 10.0
            self.current_zone += 1
            if self.current_zone > self.FINAL_ZONE:
                self.game_over = True # Win
            else:
                self._generate_zone()
                self.new_zone_transition = 1.0
                # sfx: new_zone.wav
                # Unlock new patterns/increase difficulty
                if self.current_zone == 3: self.unlocked_patterns.add(2)
                if self.current_zone == 5: self.unlocked_patterns.add(3)

        if not teleported: # Penalty for not moving on beat
            self.is_camouflaged = False

        self._update_valid_targets()
        self.selected_target_idx = -1
        
        return reward

    def _update_continuous_logic(self):
        """Handles logic that runs every frame, not just on beats."""
        # Interpolate player position for smooth animation
        interp_factor = self.beat_timer / self.beat_duration
        self.player_render_pos = self._lerp_pos(
            self._grid_to_pixel(self.player_last_grid_pos),
            self._grid_to_pixel(self.player_grid_pos),
            interp_factor
        )
        # Update new zone transition
        if self.new_zone_transition > 0:
            self.new_zone_transition = max(0, self.new_zone_transition - 0.05)
        
        # Update distractions
        for dist in self.distractions:
            dist['timer'] -= 1
        self.distractions = [d for d in self.distractions if d['timer'] > 0]
        
        return 0.0
    
    def _update_effects(self):
        """Update particle animations."""
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        while self.particles and self.particles[0]['life'] <= 0:
            self.particles.popleft()

    def _generate_zone(self):
        """Creates a new level layout, placing tiles, guards, and the player."""
        self.grid = [[{'pattern': 0, 'is_chain_reaction': False} for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        
        num_patterns = len(self.unlocked_patterns) + 1
        
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.np_random.random() > 0.3:
                    self.grid[y][x]['pattern'] = self.np_random.choice(list(self.unlocked_patterns))
                else:
                    self.grid[y][x]['pattern'] = 0 # Rock/unsafe
                
                # Add a chance for a chain reaction tile
                if self.current_zone >= 2 and self.np_random.random() < 0.05:
                     self.grid[y][x]['is_chain_reaction'] = True

        self.player_grid_pos = (1, self.GRID_HEIGHT // 2)
        self.exit_pos = (self.GRID_WIDTH - 2, self.np_random.integers(1, self.GRID_HEIGHT - 1))
        
        self.grid[self.player_grid_pos[1]][self.player_grid_pos[0]]['pattern'] = 1
        self.grid[self.exit_pos[1]][self.exit_pos[0]]['pattern'] = 1
        
        # Add guards
        self.guards = []
        num_guards = min(1 + self.current_zone, 5)
        guard_speed = 1.0 + (self.current_zone // 2) * 0.1 # tiles per beat
        for i in range(num_guards):
            path_type = self.np_random.choice(['horizontal', 'vertical', 'box'])
            if path_type == 'horizontal':
                y_pos = self.np_random.integers(0, self.GRID_HEIGHT)
                path = [(x, y_pos) for x in range(2, self.GRID_WIDTH - 2)]
            elif path_type == 'vertical':
                x_pos = self.np_random.integers(0, self.GRID_WIDTH)
                path = [(x_pos, y) for y in range(2, self.GRID_HEIGHT - 2)]
            else: # box
                x1, y1 = self.np_random.integers(2, self.GRID_WIDTH-4), self.np_random.integers(2, self.GRID_HEIGHT-4)
                x2, y2 = x1 + self.np_random.integers(2,4), y1 + self.np_random.integers(2,4)
                path = [(x, y1) for x in range(x1, x2)] + [(x2, y) for y in range(y1, y2)] + \
                       [(x, y2) for x in range(x2, x1, -1)] + [(x1, y) for y in range(y2, y1, -1)]

            if not path: continue
            self.guards.append({
                'path': path, 'path_idx': 0, 'pos': path[0], 'last_pos': path[0],
                'dir': (1, 0), 'speed': guard_speed, 'distraction_timer': 0
            })

    def _update_guards(self):
        for guard in self.guards:
            if guard['distraction_timer'] > 0:
                guard['distraction_timer'] -= 1
                # Guard is distracted, doesn't move
                guard['last_pos'] = guard['pos']
                continue

            # Move along path
            current_pos = guard['path'][guard['path_idx']]
            
            # Check for distractions
            for dist in self.distractions:
                dist_sq = (current_pos[0] - dist['pos'][0])**2 + (current_pos[1] - dist['pos'][1])**2
                if dist_sq < 25: # If within 5 tiles
                    guard['distraction_timer'] = 5 # Distracted for 5 beats
                    guard['distraction_target'] = dist['pos']
                    break
            
            if guard['distraction_timer'] > 0: # Just got distracted
                # Move towards distraction target
                dx = guard['distraction_target'][0] - current_pos[0]
                dy = guard['distraction_target'][1] - current_pos[1]
                if abs(dx) > abs(dy):
                    next_pos = (current_pos[0] + np.sign(dx), current_pos[1])
                else:
                    next_pos = (current_pos[0], current_pos[1] + np.sign(dy))
                if 0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT:
                    guard['dir'] = (np.sign(dx), np.sign(dy))
                    guard['last_pos'] = guard['pos']
                    guard['pos'] = next_pos
                continue

            # Normal pathing
            next_idx = (guard['path_idx'] + 1) % len(guard['path'])
            next_pos = guard['path'][next_idx]
            
            guard['dir'] = (next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])
            guard['last_pos'] = guard['pos']
            guard['pos'] = next_pos
            guard['path_idx'] = next_idx

    def _update_detection(self):
        """Checks if player is seen by guards and updates detection level."""
        is_visible = False
        if not self.is_camouflaged:
            player_tile = self.grid[self.player_grid_pos[1]][self.player_grid_pos[0]]
            if player_tile['pattern'] != self.player_pattern:
                is_visible = True
        
        seen_this_frame = False
        for guard in self.guards:
            g_pos = guard['pos']
            p_pos = self.player_grid_pos
            dist_sq = (g_pos[0] - p_pos[0])**2 + (g_pos[1] - p_pos[1])**2
            
            if dist_sq < 8**2: # Detection radius
                vec_to_player = (p_pos[0] - g_pos[0], p_pos[1] - g_pos[1])
                guard_dir = guard['dir']
                
                # Normalize vectors
                len_v = math.sqrt(vec_to_player[0]**2 + vec_to_player[1]**2)
                if len_v == 0: continue
                vec_to_player_norm = (vec_to_player[0]/len_v, vec_to_player[1]/len_v)
                
                len_g = math.sqrt(guard_dir[0]**2 + guard_dir[1]**2)
                if len_g == 0: continue
                guard_dir_norm = (guard_dir[0]/len_g, guard_dir[1]/len_g)

                dot_product = vec_to_player_norm[0] * guard_dir_norm[0] + vec_to_player_norm[1] * guard_dir_norm[1]
                
                if dot_product > math.cos(math.radians(45)): # 90 degree cone
                    seen_this_frame = True
                    break
        
        if seen_this_frame and is_visible:
            self.detection_level += 0.4 # Fast detection
            # sfx: detection_increase.wav
        else:
            self.detection_level = max(0, self.detection_level - 0.1)

        if self.detection_level >= 1.0:
            self.game_over = True
            # sfx: game_over_detected.wav
            return -50.0 # Detection penalty
        
        if seen_this_frame and is_visible:
             return -0.1 # Penalty for being in sight line while visible

        return 0.0

    def _trigger_chain_reaction(self, pos):
        """Creates a distraction at a nearby location."""
        distraction_pos = (pos[0] + self.np_random.choice([-2, 2]), pos[1] + self.np_random.choice([-2, 2]))
        distraction_pos = (
            max(0, min(self.GRID_WIDTH - 1, distraction_pos[0])),
            max(0, min(self.GRID_HEIGHT - 1, distraction_pos[1]))
        )
        self.distractions.append({'pos': distraction_pos, 'timer': 150}) # lasts 5 beats
        self._create_particles(self._grid_to_pixel(distraction_pos), self.CHAIN_REACTION_COLOR, 30)

    def _update_valid_targets(self):
        """Finds all tiles the player can teleport to."""
        self.valid_targets = []
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if (x, y) == self.player_grid_pos:
                    continue
                tile = self.grid[y][x]
                if tile['pattern'] in self.unlocked_patterns:
                    self.valid_targets.append((x, y))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "zone": self.current_zone}
    
    # --- Rendering Methods ---
    
    def _render_game(self):
        # Beat Pulse
        beat_progress = (self.beat_timer / self.beat_duration)
        pulse_size = int(self.TILE_SIZE * (1 - beat_progress) * 0.5)
        pulse_alpha = int(50 * (1 - beat_progress))
        if pulse_alpha > 0:
            center = (self.SCREEN_WIDTH//2, self.SCREEN_HEIGHT//2)
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], self.SCREEN_WIDTH//2 - pulse_size, (20, 40, 70, pulse_alpha))
            
        # Grid and Tiles
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                tile = self.grid[y][x]
                color = self.PATTERN_COLORS[tile['pattern']]
                pygame.draw.rect(self.screen, color, rect)
                if tile['is_chain_reaction']:
                    pygame.draw.circle(self.screen, self.CHAIN_REACTION_COLOR, rect.center, 5)
                # Draw grid lines
                pygame.draw.rect(self.screen, self.COLOR_BG, rect, 1)

        # Valid Targets
        for i, pos in enumerate(self.valid_targets):
            center = self._grid_to_pixel(pos)
            color = self.COLOR_TARGET_SELECTED if i == self.selected_target_idx else self.COLOR_TARGET_VALID
            pygame.draw.circle(self.screen, color, center, 5, 2)

        # Guards
        for guard in self.guards:
            interp_factor = self.beat_timer / self.beat_duration
            render_pos = self._lerp_pos(self._grid_to_pixel(guard['last_pos']), self._grid_to_pixel(guard['pos']), interp_factor)
            
            # Vision Cone
            self._draw_vision_cone(render_pos, guard['dir'])
            
            # Body
            self._draw_glow_circle(render_pos, 12, self.COLOR_GUARD, 40)

        # Distractions
        for dist in self.distractions:
            alpha = min(255, dist['timer'] * 5)
            color = (*self.CHAIN_REACTION_COLOR, alpha)
            pos = self._grid_to_pixel(dist['pos'])
            radius = int(10 + 10 * math.sin(self.steps * 0.2))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            if alpha > 0:
                color = (*p['color'], alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size']), color)

        # Player
        player_color = self.COLOR_PLAYER if self.is_camouflaged else (255, 100, 100)
        self._draw_glow_circle(self.player_render_pos, 15, player_color, 60)
        
        # New Zone Transition
        if self.new_zone_transition > 0:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            alpha = int(255 * self.new_zone_transition)
            overlay.fill((255, 255, 255, alpha))
            self.screen.blit(overlay, (0, 0))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Zone
        zone_text = self.font_large.render(f"ZONE: {self.current_zone}/{self.FINAL_ZONE}", True, self.COLOR_TEXT)
        self.screen.blit(zone_text, (self.SCREEN_WIDTH - zone_text.get_width() - 10, 10))

        # Detection Meter
        bar_width, bar_height = 200, 20
        bar_x, bar_y = (self.SCREEN_WIDTH - bar_width) // 2, self.SCREEN_HEIGHT - bar_height - 10
        pygame.draw.rect(self.screen, (80, 80, 80), (bar_x, bar_y, bar_width, bar_height))
        fill_width = bar_width * self.detection_level
        pygame.draw.rect(self.screen, self.COLOR_GUARD, (bar_x, bar_y, fill_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (bar_x, bar_y, bar_width, bar_height), 2)
        det_text = self.font_small.render("DETECTION", True, self.COLOR_TEXT)
        self.screen.blit(det_text, (bar_x + (bar_width - det_text.get_width())//2, bar_y + 2))

    # --- Helper & Utility Methods ---

    def _grid_to_pixel(self, grid_pos):
        x = int(grid_pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2)
        y = int(grid_pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2)
        return (x, y)

    def _lerp(self, a, b, t):
        return a + (b - a) * t

    def _lerp_pos(self, p1, p2, t):
        return (self._lerp(p1[0], p2[0], t), self._lerp(p1[1], p2[1], t))

    def _draw_glow_circle(self, pos, radius, color, glow_strength):
        pos_int = (int(pos[0]), int(pos[1]))
        for i in range(radius, 0, -1):
            alpha = int(glow_strength * (1 - (i / radius))**2)
            if alpha > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], i, (*color, alpha))
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius // 2, color)
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius // 2, color)

    def _draw_vision_cone(self, pos, direction):
        pos_int = (int(pos[0]), int(pos[1]))
        if direction == (0,0): return
        
        angle = math.atan2(direction[1], direction[0])
        cone_angle = math.pi / 2 # 90 degrees
        cone_len = 6 * self.TILE_SIZE

        p1 = pos_int
        p2 = (pos_int[0] + cone_len * math.cos(angle - cone_angle / 2),
              pos_int[1] + cone_len * math.sin(angle - cone_angle / 2))
        p3 = (pos_int[0] + cone_len * math.cos(angle + cone_angle / 2),
              pos_int[1] + cone_len * math.sin(angle + cone_angle / 2))

        pygame.gfxdraw.filled_trigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_GUARD_CONE)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 30),
                'max_life': 30,
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

if __name__ == '__main__':
    # --- Manual Play Script ---
    # This part is for local testing and visualization.
    # It will not be run by the evaluation server.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Bioluminescent Cephalopod")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    movement = 0
    space_held = 0
    shift_held = 0
    
    print("--- Controls ---")
    print(GameEnv.user_guide)
    print("R: Reset environment")
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                if event.key == pygame.K_SPACE: space_held = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE: space_held = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 0

        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}, Info: {info}")
            # Reset on finish to play again
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(30) # Run at 30 FPS
        
    pygame.quit()