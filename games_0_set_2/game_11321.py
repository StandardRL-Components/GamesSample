import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:50:48.656675
# Source Brief: brief_01321.md
# Brief Index: 1321
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a stealth/puzzle/rhythm game.
    The agent navigates a fractal landscape, evading patrols by crafting
    distractions in time with a global rhythm.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a fractal landscape and evade patrols. Craft distractions to slip by undetected and reach the exit."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to cycle through craftable distractions and shift to deploy one."
    )
    auto_advance = True

    # --- Colors ---
    COLOR_BG = (15, 10, 25)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255, 50)
    COLOR_PATROL = (255, 50, 50)
    COLOR_PATROL_GLOW = (255, 50, 50, 70)
    COLOR_EXIT = (255, 255, 0)
    COLOR_EXIT_GLOW = (255, 255, 0, 80)
    COLOR_WORD = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_RHYTHM_BAR = (100, 100, 150)
    COLOR_RHYTHM_PULSE = (200, 200, 255)
    
    # --- Game Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 2000
    PLAYER_SPEED = 3.0
    PLAYER_RADIUS = 8
    PATROL_BASE_SPEED = 0.5
    PATROL_MAX_SPEED = 2.0
    PATROL_DETECTION_RADIUS = 30
    PATROL_NEAR_MISS_MULTIPLIER = 2.5
    RHYTHM_PERIOD = 120 # steps for a full cycle

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 16, bold=True)
        self.large_font = pygame.font.SysFont("monospace", 48, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.exit_pos = None
        self.patrols = []
        self.crafted_words = []
        self.particles = []
        
        # Action state
        self.prev_space_held = False
        self.prev_shift_held = False

        # Progression state
        self.consecutive_evades = 0
        self.word_patterns = []
        self.unlocked_pattern_count = 1
        self.selected_pattern_index = 0
        
        self.background_surface = pygame.Surface((self.WIDTH, self.HEIGHT))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Core State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = pygame.math.Vector2(50, self.HEIGHT / 2)
        self.exit_pos = pygame.math.Vector2(self.WIDTH - 50, self.HEIGHT / 2)

        self.crafted_words.clear()
        self.particles.clear()
        
        self.prev_space_held = False
        self.prev_shift_held = False

        # --- Reset Progression ---
        self.consecutive_evades = 0
        if options and options.get("reset_unlocks", False):
            self.unlocked_pattern_count = 1
        self._update_word_patterns()
        self.selected_pattern_index = 0

        # --- Procedural Generation ---
        self._generate_background()
        self._generate_patrols()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # According to the Gymnasium API, calling step() after an episode is
            # done is undefined behavior. Common practice is to return the last
            # observation and set flags appropriately.
            obs = self._get_observation()
            # We can't know if the last state was terminated or truncated without
            # storing it, so we default to a safe return.
            return obs, 0, self.game_over, True, self._get_info()

        self.steps += 1
        reward = 0.01  # Small reward for surviving a step

        # --- Process Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward += self._handle_actions(movement, space_held, shift_held)

        # --- Update Game Logic ---
        self._update_player(movement)
        patrol_update_info = self._update_patrols()
        self._update_words()
        self._update_particles()
        
        # --- Update Rewards & State from Updates ---
        reward += patrol_update_info["reward"]
        if patrol_update_info["new_unlock"]:
             reward += 10.0
             self.unlocked_pattern_count += 1
             self._update_word_patterns()

        # --- Check Termination and Truncation ---
        terminated, term_reward = self._check_termination()
        reward += term_reward

        truncated = self.steps >= self.MAX_STEPS
        
        self.game_over = terminated or truncated
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_actions(self, movement, space_held, shift_held):
        reward = 0
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held

        if space_press:
            # Select next fractal pattern
            self.selected_pattern_index = (self.selected_pattern_index + 1) % len(self.word_patterns)
            # sfx: UI_SELECT

        if shift_press and self.word_patterns:
            # Craft the currently selected fractal word
            pattern = self.word_patterns[self.selected_pattern_index]
            new_word = {
                "pos": self.player_pos.copy(),
                "timer": pattern["duration"],
                "max_timer": pattern["duration"],
                "radius": pattern["radius"],
                "color": pattern["color"],
                "distracted_patrols": set()
            }
            self.crafted_words.append(new_word)
            # sfx: WORD_CRAFT
            self._spawn_particles(self.player_pos, 20, pattern['color'])

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        return reward

    def _update_player(self, movement):
        velocity = pygame.math.Vector2(0, 0)
        if movement == 1: velocity.y = -1 # Up
        elif movement == 2: velocity.y = 1 # Down
        elif movement == 3: velocity.x = -1 # Left
        elif movement == 4: velocity.x = 1 # Right
        
        if velocity.length() > 0:
            velocity.normalize_ip()
            self.player_pos += velocity * self.PLAYER_SPEED

        # Clamp player position to screen bounds
        self.player_pos.x = max(self.PLAYER_RADIUS, min(self.player_pos.x, self.WIDTH - self.PLAYER_RADIUS))
        self.player_pos.y = max(self.PLAYER_RADIUS, min(self.player_pos.y, self.HEIGHT - self.PLAYER_RADIUS))

    def _update_patrols(self):
        # Difficulty scaling
        speed_increase = 0.05 * (self.steps // 200)
        current_patrol_speed = min(self.PATROL_BASE_SPEED + speed_increase, self.PATROL_MAX_SPEED)
        
        distraction_reward = 0
        new_unlock = False

        for p in self.patrols:
            # --- Check for Distraction ---
            is_distracted = False
            for word in self.crafted_words:
                if p['pos'].distance_to(word['pos']) < word['radius']:
                    is_distracted = True
                    if p['id'] not in word['distracted_patrols']:
                         distraction_reward += 5.0 # Reward for new distraction
                         word['distracted_patrols'].add(p['id'])
                    break
            p['is_distracted'] = is_distracted
            
            # --- Movement ---
            if not is_distracted:
                target_pos = p['path'][p['target_idx']]
                direction = target_pos - p['pos']
                if direction.length() < current_patrol_speed:
                    p['pos'] = target_pos
                    p['target_idx'] = (p['target_idx'] + 1) % len(p['path'])
                else:
                    p['pos'] += direction.normalize() * current_patrol_speed
            
            # --- Near Miss Check for Unlocks ---
            dist_to_player = p['pos'].distance_to(self.player_pos)
            near_miss_radius = self.PATROL_DETECTION_RADIUS * self.PATROL_NEAR_MISS_MULTIPLIER
            
            is_near = dist_to_player < near_miss_radius
            if is_near and not p['was_near']:
                self.consecutive_evades += 1
                if self.consecutive_evades >= 5:
                    self.consecutive_evades = 0
                    if self.unlocked_pattern_count < len(self._get_all_possible_patterns()):
                        new_unlock = True
                        # sfx: UNLOCK_PATTERN
            p['was_near'] = is_near

        return {"reward": distraction_reward, "new_unlock": new_unlock}

    def _update_words(self):
        self.crafted_words = [w for w in self.crafted_words if w['timer'] > 0]
        for w in self.crafted_words:
            w['timer'] -= 1

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['alpha'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['alpha'] -= 5
            p['vel'] *= 0.98 # Friction

    def _check_termination(self):
        # Win condition: reached exit
        if self.player_pos.distance_to(self.exit_pos) < self.PLAYER_RADIUS + 10:
            # sfx: WIN
            return True, 100.0

        # Lose condition: detected by patrol
        for p in self.patrols:
            if not p['is_distracted']:
                if p['pos'].distance_to(self.player_pos) < self.PATROL_DETECTION_RADIUS:
                    self.consecutive_evades = 0 # Reset on detection
                    # sfx: LOSE
                    return True, -100.0
            
        return False, 0.0

    def _get_observation(self):
        # --- Render Background ---
        self.screen.blit(self.background_surface, (0, 0))

        # --- Render Game Elements ---
        self._render_exit()
        self._render_words()
        self._render_patrols()
        self._render_player()
        self._render_particles()

        # --- Render UI ---
        self._render_ui()

        # --- Game Over Screen ---
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            # Check win/loss state for text
            is_win = self.player_pos.distance_to(self.exit_pos) < self.PLAYER_RADIUS + 10
            if is_win:
                text = "SYSTEM ESCAPED"
                color = self.COLOR_EXIT
            else:
                text = "CONNECTION LOST"
                color = self.COLOR_PATROL
            
            text_surf = self.large_font.render(text, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "unlocked_patterns": self.unlocked_pattern_count,
            "consecutive_evades": self.consecutive_evades,
        }

    # =================================================================
    # --- RENDER HELPERS ---
    # =================================================================

    def _render_player(self):
        pos = (int(self.player_pos.x), int(self.player_pos.y))
        # Glow effect
        for i in range(self.PLAYER_RADIUS, self.PLAYER_RADIUS + 5):
            alpha = self.COLOR_PLAYER_GLOW[3] * (1 - (i - self.PLAYER_RADIUS) / 5)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], i, (*self.COLOR_PLAYER_GLOW[:3], int(alpha)))
        # Core
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _render_patrols(self):
        rhythm_progress = (self.steps % self.RHYTHM_PERIOD) / self.RHYTHM_PERIOD
        pulse = (math.sin(rhythm_progress * 2 * math.pi) + 1) / 2 # 0 to 1
        
        for p in self.patrols:
            pos = (int(p['pos'].x), int(p['pos'].y))
            radius = int(10 + pulse * 4)
            
            # Draw detection radius if not distracted
            if not p['is_distracted']:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PATROL_DETECTION_RADIUS, (*self.COLOR_PATROL, 80))
            else:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PATROL_DETECTION_RADIUS, (100, 255, 100, 80))

            # Glow
            glow_radius = int(radius * 1.5)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, self.COLOR_PATROL_GLOW)
            # Core shape (pulsating triangle)
            points = []
            for i in range(3):
                angle = 2 * math.pi * i / 3 + rhythm_progress * math.pi
                points.append((pos[0] + radius * math.cos(angle), pos[1] + radius * math.sin(angle)))
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PATROL)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PATROL)
    
    def _render_exit(self):
        pos = (int(self.exit_pos.x), int(self.exit_pos.y))
        radius = 15
        # Glow
        for i in range(radius, radius + 10):
            alpha = self.COLOR_EXIT_GLOW[3] * (1 - (i - radius) / 10)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], i, (*self.COLOR_EXIT_GLOW[:3], int(alpha)))
        # Core
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_EXIT)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_EXIT)

    def _render_words(self):
        for word in self.crafted_words:
            progress = word['timer'] / word['max_timer']
            current_radius = word['radius'] * (1 - progress)
            alpha = int(255 * progress)
            color = (*word['color'], alpha)
            pos = (int(word['pos'].x), int(word['pos'].y))

            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(current_radius), color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(current_radius * 0.66), color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(current_radius * 0.33), color)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            color = (*p['color'], int(p['alpha']))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 2, color)

    def _render_ui(self):
        # --- Score and Steps ---
        score_text = f"SCORE: {int(self.score)}"
        steps_text = f"STEPS: {self.steps}/{self.MAX_STEPS}"
        score_surf = self.font.render(score_text, True, self.COLOR_UI_TEXT)
        steps_surf = self.font.render(steps_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))
        self.screen.blit(steps_surf, (10, 30))

        # --- Selected Word Pattern ---
        if self.word_patterns:
            pattern_name = self.word_patterns[self.selected_pattern_index]['name']
            pattern_text = f"CRAFT: <{pattern_name}>"
            pattern_surf = self.font.render(pattern_text, True, self.COLOR_UI_TEXT)
            self.screen.blit(pattern_surf, (self.WIDTH - pattern_surf.get_width() - 10, 10))

        # --- Rhythm Bar ---
        rhythm_progress = (self.steps % self.RHYTHM_PERIOD) / self.RHYTHM_PERIOD
        pulse = (math.sin(rhythm_progress * 2 * math.pi) + 1) / 2
        
        bar_y = self.HEIGHT - 20
        bar_height = 10
        pygame.draw.rect(self.screen, self.COLOR_RHYTHM_BAR, (10, bar_y, self.WIDTH - 20, bar_height), 1)
        
        pulse_width = (self.WIDTH - 20) * pulse
        pulse_x = 10 + ((self.WIDTH - 20) - pulse_width) / 2
        pygame.draw.rect(self.screen, self.COLOR_RHYTHM_PULSE, (pulse_x, bar_y, pulse_width, bar_height))

    # =================================================================
    # --- GENERATION HELPERS ---
    # =================================================================

    def _generate_background(self):
        self.background_surface.fill(self.COLOR_BG)
        for i in range(5):
            start_x = self.np_random.uniform(0, self.WIDTH)
            start_len = self.np_random.uniform(40, 80)
            self._draw_fractal_branch(
                self.background_surface, 
                pygame.math.Vector2(start_x, self.HEIGHT), 
                -math.pi / 2, 
                start_len, 
                depth=self.np_random.integers(5, 8),
                color=(30, 20, 50),
                width=8
            )

    def _draw_fractal_branch(self, surface, start_pos, angle, length, depth, color, width):
        if depth <= 0 or length < 2:
            return
        
        end_pos = start_pos + pygame.math.Vector2(math.cos(angle), math.sin(angle)) * length
        pygame.draw.line(surface, color, start_pos, end_pos, max(1, int(width)))

        new_length = length * 0.75
        new_width = width * 0.7
        angle_diff = self.np_random.uniform(0.3, 0.6) # radians

        self._draw_fractal_branch(surface, end_pos, angle - angle_diff, new_length, depth - 1, color, new_width)
        self._draw_fractal_branch(surface, end_pos, angle + angle_diff, new_length, depth - 1, color, new_width)

    def _generate_patrols(self):
        self.patrols.clear()
        num_patrols = self.np_random.integers(2, 5)
        for i in range(num_patrols):
            path_type = self.np_random.choice(['box', 'line'])
            path = []
            if path_type == 'box':
                x1, y1 = self.np_random.uniform(100, self.WIDTH - 100), self.np_random.uniform(50, self.HEIGHT - 50)
                w, h = self.np_random.uniform(80, 200), self.np_random.uniform(80, 200)
                x2, y2 = x1 + w, y1 + h
                path = [pygame.math.Vector2(x1, y1), pygame.math.Vector2(x2, y1), pygame.math.Vector2(x2, y2), pygame.math.Vector2(x1, y2)]
            elif path_type == 'line':
                x1, y1 = self.np_random.uniform(100, self.WIDTH - 100), self.np_random.uniform(50, self.HEIGHT - 50)
                x2, y2 = self.np_random.uniform(100, self.WIDTH - 100), self.np_random.uniform(50, self.HEIGHT - 50)
                path = [pygame.math.Vector2(x1, y1), pygame.math.Vector2(x2, y2)]
            
            self.patrols.append({
                'id': i,
                'path': path,
                'pos': path[0].copy(),
                'target_idx': 1,
                'is_distracted': False,
                'was_near': False,
            })
    
    def _get_all_possible_patterns(self):
        return [
            {'name': 'Pulse', 'radius': 75, 'duration': 90, 'color': (255, 255, 255)},
            {'name': 'Wave', 'radius': 120, 'duration': 60, 'color': (200, 200, 255)},
            {'name': 'Burst', 'radius': 50, 'duration': 150, 'color': (255, 255, 200)},
            {'name': 'Echo', 'radius': 150, 'duration': 120, 'color': (200, 255, 200)},
        ]

    def _update_word_patterns(self):
        all_patterns = self._get_all_possible_patterns()
        self.word_patterns = all_patterns[:self.unlocked_pattern_count]
    
    def _spawn_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'alpha': 255,
                'color': color
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        obs, _ = self.reset()
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
    # This block allows you to run the file directly to play the game
    # We need to switch the video driver to play manually
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Control Mapping ---
    # Arrows: Move
    # Space: Select next pattern
    # Left Shift: Craft word
    
    action = [0, 0, 0] # [movement, space, shift]
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Fractal Stealth")
    clock = pygame.time.Clock()
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # --- Keyboard Input for Manual Play ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # --- Render Observation ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(60) # Control render speed

    print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
    env.close()