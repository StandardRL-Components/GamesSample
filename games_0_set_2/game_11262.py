import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:52:14.621153
# Source Brief: brief_01262.md
# Brief Index: 1262
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: Steal valuable musical artifacts by matching musical notes to unlock
    combo moves that disable security systems and bypass guards.
    A stealth puzzle game with rhythm elements, designed for visual quality and gameplay feel.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Infiltrate a high-security vault in this stealth puzzle game. "
        "Collect musical notes to form combos, disable lasers, and stun guards to steal the valuable artifact."
    )
    user_guide = (
        "Controls: Use ↑↓←→ arrow keys to move. Press Shift to cycle through unlocked combos and Space to activate the selected one."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.UI_HEIGHT = 60

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        # --- Visual & Style ---
        self._define_colors()
        self.font_main = pygame.font.Font(None, 24)
        self.font_big = pygame.font.Font(None, 36)
        self.font_combo = pygame.font.Font(None, 20)

        # --- Game Parameters ---
        self.MAX_STEPS = 5000
        self.PLAYER_SPEED = 12
        self.GUARD_BASE_SPEED = 1.0
        self.GUARD_BASE_VISION_RANGE = 90
        self.LERP_FACTOR = 0.5  # For smooth rendering

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0.0
        self.player_pos = pygame.Vector2(0, 0)
        self.player_render_pos = pygame.Vector2(0, 0)
        self.guards = []
        self.walls = []
        self.lasers = []
        self.note_pads = []
        self.player_note_sequence = []
        self.available_combos = {}
        self.unlocked_combos = []
        self.selected_combo_index = 0
        self.artifact_pos = pygame.Vector2(0, 0)
        self.artifact_collected = False
        self.exit_pos = pygame.Vector2(0, 0)
        self.alarm_triggered = False
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []

        self.reset()
        # self.validate_implementation() # Optional self-check

    def _define_colors(self):
        """Define the game's color palette for a futuristic art deco style."""
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_WALL = (60, 80, 100)
        self.COLOR_WALL_ACCENT = (80, 100, 130)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_PLAYER_GLOW = (0, 255, 255, 50)
        self.COLOR_GUARD = (255, 165, 0)
        self.COLOR_GUARD_VISION = (255, 165, 0, 40)
        self.COLOR_GUARD_STUNNED = (100, 100, 100)
        self.COLOR_LASER_DANGER = (255, 20, 20)
        self.COLOR_LASER_SAFE = (50, 255, 50, 50)
        self.COLOR_ARTIFACT = (255, 223, 0)
        self.COLOR_EXIT = (144, 238, 144)
        self.COLOR_UI_BG = (10, 20, 30)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.NOTE_COLORS = {
            1: (255, 80, 80),  # Red
            2: (80, 255, 80),  # Green
            3: (80, 80, 255),  # Blue
            4: (255, 255, 80)  # Yellow
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0.0
        self.alarm_triggered = False
        self.artifact_collected = False

        self._setup_level()

        self.player_pos = pygame.Vector2(self.start_pos)
        self.player_render_pos = pygame.Vector2(self.player_pos)
        self.player_note_sequence = []

        self.unlocked_combos = []
        self.selected_combo_index = 0

        self.prev_space_held = False
        self.prev_shift_held = False

        self.particles = []

        return self._get_observation(), self._get_info()

    def _setup_level(self):
        """Initializes the layout, entities, and puzzle elements for the level."""
        game_area_height = self.SCREEN_HEIGHT - self.UI_HEIGHT
        self.start_pos = pygame.Vector2(50, game_area_height / 2)
        self.exit_pos = pygame.Vector2(self.SCREEN_WIDTH - 50, game_area_height / 2)
        self.artifact_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, 50)

        self.walls = [
            pygame.Rect(0, 0, self.SCREEN_WIDTH, 10),
            pygame.Rect(0, game_area_height - 10, self.SCREEN_WIDTH, 10),
            pygame.Rect(0, 0, 10, game_area_height),
            pygame.Rect(self.SCREEN_WIDTH - 10, 0, 10, game_area_height),
            pygame.Rect(150, 10, 20, 150),
            pygame.Rect(self.SCREEN_WIDTH - 170, game_area_height - 160, 20, 150),
            pygame.Rect(250, 200, 140, 20)
        ]

        self.lasers = [
            {'p1': pygame.Vector2(250, 10), 'p2': pygame.Vector2(250, 200), 'active': True, 'cycle_time': 120, 'timer': 0},
            {'p1': pygame.Vector2(390, 220), 'p2': pygame.Vector2(390, game_area_height), 'active': True, 'cycle_time': 180, 'timer': 90}
        ]

        self.note_pads = [
            {'pos': pygame.Vector2(100, 100), 'id': 1, 'radius': 10},
            {'pos': pygame.Vector2(100, 250), 'id': 2, 'radius': 10},
            {'pos': pygame.Vector2(200, 280), 'id': 3, 'radius': 10},
        ]
        
        self.available_combos = {
            'DISABLE_LASER': {'name': 'EMP Pulse', 'seq': (1, 2)},
            'STUN_GUARD': {'name': 'Sonic Burst', 'seq': (1, 3, 2)},
        }

        self.guards = [
            {'pos': pygame.Vector2(450, 100), 'render_pos': pygame.Vector2(450, 100), 'path': [pygame.Vector2(450, 100), pygame.Vector2(550, 100), pygame.Vector2(550, 250), pygame.Vector2(450, 250)], 'waypoint_idx': 0, 'speed': self.GUARD_BASE_SPEED, 'vision_range': self.GUARD_BASE_VISION_RANGE, 'stun_timer': 0}
        ]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.reward_this_step = 0.0

        dist_to_artifact_before = self.player_pos.distance_to(self.artifact_pos)
        min_dist_to_guard_before = self._get_min_dist_to_guard()

        self._handle_input(action)
        self._update_guards()
        self._update_lasers()
        self._update_player_state()
        self._update_difficulty()
        self._update_particles()

        dist_to_artifact_after = self.player_pos.distance_to(self.artifact_pos)
        min_dist_to_guard_after = self._get_min_dist_to_guard()
        
        # Continuous rewards
        if not self.artifact_collected:
            self.reward_this_step += (dist_to_artifact_before - dist_to_artifact_after) * 0.01
        if min_dist_to_guard_before > 0 and min_dist_to_guard_after > 0:
            self.reward_this_step += (min_dist_to_guard_after - min_dist_to_guard_before) * 0.01

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
            self.reward_this_step -= 10.0 # Penalty for timeout
        
        self.score += self.reward_this_step
        
        return self._get_observation(), self.reward_this_step, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Handle movement
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1  # Up
        elif movement == 2: move_vec.y = 1   # Down
        elif movement == 3: move_vec.x = -1  # Left
        elif movement == 4: move_vec.x = 1   # Right
        
        if move_vec.length() > 0:
            new_pos = self.player_pos + move_vec * self.PLAYER_SPEED
            player_rect = pygame.Rect(new_pos.x - 5, new_pos.y - 5, 10, 10)
            
            can_move = True
            for wall in self.walls:
                if wall.colliderect(player_rect):
                    can_move = False
                    break
            
            if can_move:
                game_area_height = self.SCREEN_HEIGHT - self.UI_HEIGHT
                new_pos.x = np.clip(new_pos.x, 5, self.SCREEN_WIDTH - 5)
                new_pos.y = np.clip(new_pos.y, 5, game_area_height - 5)
                self.player_pos = new_pos

        # Handle actions (on press)
        if shift_held and not self.prev_shift_held and len(self.unlocked_combos) > 0:
            self.selected_combo_index = (self.selected_combo_index + 1) % len(self.unlocked_combos)
            # Sound: UI_switch.wav
        
        if space_held and not self.prev_space_held and len(self.unlocked_combos) > 0:
            self._activate_combo()
            # Sound: combo_activate.wav

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _activate_combo(self):
        combo_name = self.unlocked_combos.pop(self.selected_combo_index)
        self.selected_combo_index = 0
        self._create_particles(self.player_pos, self.COLOR_PLAYER, 20, 2.0)

        if combo_name == 'DISABLE_LASER':
            # Find closest laser and disable it
            closest_laser = None
            min_dist = float('inf')
            for laser in self.lasers:
                if laser['active']:
                    # Simplified distance check to laser midpoint
                    mid_point = laser['p1'].lerp(laser['p2'], 0.5)
                    dist = self.player_pos.distance_to(mid_point)
                    if dist < min_dist:
                        min_dist = dist
                        closest_laser = laser
            if closest_laser:
                closest_laser['active'] = False
                self.reward_this_step += 2.0
                # Sound: laser_disable.wav

        elif combo_name == 'STUN_GUARD':
            # Find closest guard and stun it
            closest_guard = None
            min_dist = float('inf')
            for guard in self.guards:
                dist = self.player_pos.distance_to(guard['pos'])
                if dist < min_dist:
                    min_dist = dist
                    closest_guard = guard
            if closest_guard and min_dist < 150: # Stun has range
                closest_guard['stun_timer'] = 150 # Stun for 150 steps
                self.reward_this_step += 1.0 # Reward for successful bypass/stun
                # Sound: guard_stun.wav

    def _update_guards(self):
        for guard in self.guards:
            if guard['stun_timer'] > 0:
                guard['stun_timer'] -= 1
                continue

            target_waypoint = guard['path'][guard['waypoint_idx']]
            direction = (target_waypoint - guard['pos'])
            
            if direction.length() < guard['speed']:
                guard['pos'] = target_waypoint
                guard['waypoint_idx'] = (guard['waypoint_idx'] + 1) % len(guard['path'])
            else:
                guard['pos'] += direction.normalize() * guard['speed']

    def _update_lasers(self):
        for laser in self.lasers:
            # Lasers that were disabled by combos stay disabled
            if not laser['active']:
                continue
            # Other lasers can cycle
            if laser['cycle_time'] > 0:
                laser['timer'] = (laser['timer'] + 1) % (laser['cycle_time'] * 2)
                laser['active'] = laser['timer'] < laser['cycle_time']


    def _update_player_state(self):
        # Check for note pad collision
        for pad in self.note_pads:
            if self.player_pos.distance_to(pad['pos']) < pad['radius'] + 5:
                if not self.player_note_sequence or self.player_note_sequence[-1] != pad['id']:
                    self.player_note_sequence.append(pad['id'])
                    self._create_particles(pad['pos'], self.NOTE_COLORS[pad['id']], 10, 1.0)
                    # Sound: note_collect.wav
                    # Check for combo match
                    for name, combo_data in self.available_combos.items():
                        if tuple(self.player_note_sequence) == combo_data['seq']:
                            if name not in self.unlocked_combos:
                                self.unlocked_combos.append(name)
                                self.reward_this_step += 5.0
                                # Sound: combo_unlocked.wav
                            self.player_note_sequence = [] # Reset sequence on match
                            break
                    if len(self.player_note_sequence) > 4: # Limit sequence length
                        self.player_note_sequence.pop(0)

        # Check for artifact collection
        if not self.artifact_collected and self.player_pos.distance_to(self.artifact_pos) < 15:
            self.artifact_collected = True
            self.reward_this_step += 20 # Intermediate reward before exit
            self._create_particles(self.artifact_pos, self.COLOR_ARTIFACT, 50, 3.0)
            # Sound: artifact_get.wav
            
    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 500 == 0:
            for guard in self.guards:
                guard['vision_range'] *= 1.05
        if self.steps > 0 and self.steps % 1000 == 0:
            for guard in self.guards:
                guard['speed'] += 0.05

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _check_termination(self):
        # Win condition
        if self.artifact_collected and self.player_pos.distance_to(self.exit_pos) < 15:
            self.game_over = True
            self.reward_this_step += 100.0
            return True

        # Lose conditions
        if self._is_player_caught():
            self.game_over = True
            self.alarm_triggered = True
            self.reward_this_step -= 50.0
            return True

        return False

    def _is_player_caught(self):
        # Laser collision
        for laser in self.lasers:
            if laser['active']:
                if self._point_segment_dist(self.player_pos, laser['p1'], laser['p2']) < 5:
                    # Sound: alarm.wav
                    return True
        
        # Guard vision
        for guard in self.guards:
            if guard['stun_timer'] > 0:
                continue
            if self.player_pos.distance_to(guard['pos']) < guard['vision_range']:
                if self._is_in_line_of_sight(guard['pos'], self.player_pos):
                    # Sound: guard_alert.wav
                    return True
        return False

    def _get_min_dist_to_guard(self):
        if not self.guards:
            return float('inf')
        return min(self.player_pos.distance_to(g['pos']) for g in self.guards)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._interpolate_positions()
        
        # Floor details
        pygame.draw.circle(self.screen, self.COLOR_EXIT, self.exit_pos, 15, 2)
        if self.artifact_collected:
             pygame.gfxdraw.filled_circle(self.screen, int(self.exit_pos.x), int(self.exit_pos.y), 13, (*self.COLOR_EXIT, 50))
        for pad in self.note_pads:
            color = self.NOTE_COLORS[pad['id']]
            pygame.gfxdraw.filled_circle(self.screen, int(pad['pos'].x), int(pad['pos'].y), pad['radius'], (*color, 100))
            pygame.gfxdraw.aacircle(self.screen, int(pad['pos'].x), int(pad['pos'].y), pad['radius'], color)

        # Walls
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)
            pygame.draw.rect(self.screen, self.COLOR_WALL_ACCENT, wall, 2)

        # Lasers
        for laser in self.lasers:
            color = self.COLOR_LASER_DANGER if laser['active'] else self.COLOR_LASER_SAFE
            if laser['active']:
                # Pulsing effect
                pulse = abs(math.sin(pygame.time.get_ticks() * 0.01))
                width = int(2 + pulse * 3)
                pygame.draw.line(self.screen, color, laser['p1'], laser['p2'], width)
            else:
                pygame.draw.line(self.screen, color, laser['p1'], laser['p2'], 1)

        # Artifact
        if not self.artifact_collected:
            t = pygame.time.get_ticks() * 0.002
            size = int(10 + 2 * math.sin(t))
            glow_size = int(size * (1.5 + 0.5 * math.sin(t*0.7)))
            pygame.gfxdraw.filled_circle(self.screen, int(self.artifact_pos.x), int(self.artifact_pos.y), glow_size, (*self.COLOR_ARTIFACT, 30))
            pygame.gfxdraw.filled_circle(self.screen, int(self.artifact_pos.x), int(self.artifact_pos.y), size, self.COLOR_ARTIFACT)

        # Guards
        for guard in self.guards:
            # Vision cone
            if guard['stun_timer'] == 0:
                self._draw_vision_cone(guard)
            # Guard body
            color = self.COLOR_GUARD_STUNNED if guard['stun_timer'] > 0 else self.COLOR_GUARD
            pos = guard['render_pos']
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), 8, color)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), 8, color)
            if guard['stun_timer'] > 0:
                # Stun effect
                angle = pygame.time.get_ticks() * 0.01
                for i in range(3):
                    sa = angle + i * 2 * math.pi / 3
                    pygame.draw.circle(self.screen, self.COLOR_UI_TEXT, (pos.x + math.cos(sa)*12, pos.y + math.sin(sa)*12), 1)

        # Player
        p_pos = self.player_render_pos
        pygame.gfxdraw.filled_circle(self.screen, int(p_pos.x), int(p_pos.y), 15, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, int(p_pos.x), int(p_pos.y), 7, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, int(p_pos.x), int(p_pos.y), 7, self.COLOR_PLAYER)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            if alpha > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['life']*0.1 + 1), color)

    def _render_ui(self):
        ui_y = self.SCREEN_HEIGHT - self.UI_HEIGHT
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, ui_y, self.SCREEN_WIDTH, self.UI_HEIGHT))
        pygame.draw.line(self.screen, self.COLOR_WALL_ACCENT, (0, ui_y), (self.SCREEN_WIDTH, ui_y), 2)

        # Score
        score_text = self.font_big.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, ui_y + 15))

        # Current sequence
        seq_text = self.font_main.render("SEQUENCE:", True, self.COLOR_UI_TEXT)
        self.screen.blit(seq_text, (200, ui_y + 10))
        for i, note_id in enumerate(self.player_note_sequence):
            pygame.draw.rect(self.screen, self.NOTE_COLORS[note_id], (200 + i * 25, ui_y + 35, 20, 15))

        # Unlocked combos
        combo_text = self.font_main.render("COMBOS:", True, self.COLOR_UI_TEXT)
        self.screen.blit(combo_text, (400, ui_y + 10))
        for i, combo_name in enumerate(self.unlocked_combos):
            is_selected = (i == self.selected_combo_index)
            color = self.COLOR_ARTIFACT if is_selected else self.COLOR_UI_TEXT
            name = self.available_combos[combo_name]['name']
            text = self.font_combo.render(f"{'> ' if is_selected else ''}{name}", True, color)
            self.screen.blit(text, (410, ui_y + 30 + i * 18))
            
        if self.alarm_triggered:
            alarm_text = self.font_big.render("ALARM!", True, self.COLOR_LASER_DANGER)
            text_rect = alarm_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 50))
            self.screen.blit(alarm_text, text_rect)

    def _interpolate_positions(self):
        self.player_render_pos.x += (self.player_pos.x - self.player_render_pos.x) * self.LERP_FACTOR
        self.player_render_pos.y += (self.player_pos.y - self.player_render_pos.y) * self.LERP_FACTOR
        for guard in self.guards:
            guard['render_pos'].x += (guard['pos'].x - guard['render_pos'].x) * self.LERP_FACTOR
            guard['render_pos'].y += (guard['pos'].y - guard['render_pos'].y) * self.LERP_FACTOR

    def _draw_vision_cone(self, guard):
        pos = guard['render_pos']
        target_pos = guard['path'][guard['waypoint_idx']]
        direction_vec = target_pos - pos
        if direction_vec.length() < 1:
            if len(guard['path']) > 1:
                next_idx = (guard['waypoint_idx'] + 1) % len(guard['path'])
                direction_vec = guard['path'][next_idx] - pos
            else:
                direction_vec = pygame.Vector2(1,0)

        angle = direction_vec.angle_to(pygame.Vector2(1, 0))
        vision_range = guard['vision_range']
        cone_angle = 45  # degrees

        points = [pos]
        for i in range(-cone_angle, cone_angle + 1, 5):
            rad = math.radians(angle + i)
            end_point = pos + pygame.Vector2(math.cos(rad), -math.sin(rad)) * vision_range
            points.append(end_point)
        
        if len(points) > 2:
            pygame.gfxdraw.filled_polygon(self.screen, [(int(p.x), int(p.y)) for p in points], self.COLOR_GUARD_VISION)

    def _is_in_line_of_sight(self, p1, p2):
        for wall in self.walls:
            if wall.clipline(p1, p2):
                return False
        return True

    def _point_segment_dist(self, p, a, b):
        if a == b: return p.distance_to(a)
        l2 = a.distance_squared_to(b)
        t = max(0, min(1, (p - a).dot(b - a) / l2))
        projection = a + t * (b - a)
        return p.distance_to(projection)

    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, max_speed)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'color': color,
                'life': random.randint(20, 40),
                'max_life': 40
            })

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "artifact_collected": self.artifact_collected}

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # Example of how to run the environment
    # This block will not run in the hosted environment but is useful for local testing.
    # To run, you might need to `pip install pygame`
    # and remove/comment out `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")`
    
    # Re-enable display for local testing
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    
    pygame.display.set_caption("Art Deco Heist")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    
    movement_action = 0  # 0=none, 1=up, 2=down, 3=left, 4=right
    space_action = 0     # 0=released, 1=held
    shift_action = 0     # 0=released, 1=held
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        else: movement_action = 0
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement_action, space_action, shift_action]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS for smooth visuals

    pygame.quit()