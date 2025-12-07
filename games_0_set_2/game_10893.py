import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:09:54.504565
# Source Brief: brief_00893.md
# Brief Index: 893
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment simulating a race through a procedurally generated quantum tunnel.
    The player must dodge quantum particles, collect energy orbs, and use card-based
    abilities to survive as long as possible and maximize their score.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Race through a procedurally generated quantum tunnel, dodging particles, collecting energy, "
        "and using card-based abilities to survive."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to activate the selected card and "
        "shift to cycle through available cards."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width, self.screen_height = 640, 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_card = pygame.font.SysFont("Consolas", 22, bold=True)

        # --- Game Constants ---
        self.PLAYER_SIZE = 10
        self.PLAYER_BASE_SPEED = 4.5
        self.PARTICLE_BASE_SPEED = 2.5
        self.TRACK_SCROLL_SPEED = 3.0
        self.TRACK_WIDTH_MIN = 150
        self.TRACK_WIDTH_MAX = 350
        self.MAX_STEPS = 5000
        self.PLAYER_TRAIL_LENGTH = 15

        # --- Visuals & Colors ---
        self.COLOR_BG = (10, 5, 25)
        self.COLOR_PLAYER = (0, 191, 255)
        self.COLOR_PLAYER_GLOW = (0, 191, 255, 60)
        self.COLOR_PARTICLE = (255, 69, 0)
        self.COLOR_PARTICLE_GLOW = (255, 69, 0, 70)
        self.COLOR_ORB = (50, 205, 50)
        self.COLOR_ORB_GLOW = (50, 205, 50, 80)
        self.COLOR_TRACK = (220, 20, 60)
        self.COLOR_TRACK_GLOW = (220, 20, 60, 50)
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_SHIELD = (148, 0, 211, 100)
        self.COLOR_SPEED_BOOST_TRAIL = (255, 215, 0)
        self.COLOR_ENERGY_BAR = (255, 255, 0)
        self.COLOR_ENERGY_BAR_BG = (50, 50, 50)

        # --- Card System (Persistent across resets) ---
        self._init_cards()
        self.unlocked_card_indices = {0, 1} # Start with Shield and Speed Boost

        # --- State Variables (initialized in reset) ---
        self.player_pos = None
        self.player_vel = None
        self.player_trail = None
        self.player_speed_modifier = 1.0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.energy = 0
        self.particle_speed = 0
        self.score_milestone_tracker = 0
        self.particles = []
        self.energy_orbs = []
        self.track_segments = []
        self.active_cards = {}
        self.current_card_index = 0
        self.last_space_state = 0
        self.last_shift_state = 0
        self.particle_spawn_timer = 0
        self.orb_spawn_timer = 0

    def _init_cards(self):
        """Defines all available cards in the game."""
        self.all_cards = [
            {'name': 'Shield', 'cost': 50, 'duration': 90},
            {'name': 'Speed Boost', 'cost': 30, 'duration': 120},
            {'name': 'Energy Magnet', 'cost': 20, 'duration': 150},
            {'name': 'Wave Clear', 'cost': 100, 'duration': 1},
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Player State
        self.player_pos = np.array([self.screen_width / 2, self.screen_height * 0.8], dtype=float)
        self.player_vel = np.array([0.0, 0.0], dtype=float)
        self.player_trail = []
        self.player_speed_modifier = 1.0

        # Game State
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.energy = 50
        self.particle_speed = self.PARTICLE_BASE_SPEED
        self.score_milestone_tracker = 0

        # Entities
        self.particles = []
        self.energy_orbs = []
        
        # Track State
        self.track_segments = []
        initial_center = self.screen_width / 2
        for y in range(self.screen_height + 50, -50, -20):
            self.track_segments.append({
                'y': float(y),
                'center_x': initial_center,
                'width': self.TRACK_WIDTH_MAX
            })

        # Card/Input State
        self.active_cards = {}
        self.current_card_index = list(self.unlocked_card_indices)[0] if self.unlocked_card_indices else 0
        self.last_space_state = 0
        self.last_shift_state = 0

        # Spawn Timers
        self.particle_spawn_timer = 60
        self.orb_spawn_timer = 45
        
        # Unlock new cards based on a fictional high score
        # For this env, we'll just add them based on a simple check
        if self.score > 500 and 2 not in self.unlocked_card_indices: self.unlocked_card_indices.add(2)
        if self.score > 1000 and 3 not in self.unlocked_card_indices: self.unlocked_card_indices.add(3)


        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.1  # Survival reward

        reward += self._handle_input(action)
        self._update_player()
        self._update_track()
        self._update_particles()
        reward += self._update_orbs()
        self._update_active_cards()
        self._spawn_entities()
        self._update_difficulty()

        if self.score // 100 > self.score_milestone_tracker:
            reward += 10 * (self.score // 100 - self.score_milestone_tracker)
            self.score_milestone_tracker = self.score // 100

        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        if self.game_over:
            reward = -100.0

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        move_vec = np.array([0.0, 0.0])
        if movement == 1: move_vec[1] = -1.0 # Up
        elif movement == 2: move_vec[1] = 1.0  # Down
        elif movement == 3: move_vec[0] = -1.0 # Left
        elif movement == 4: move_vec[0] = 1.0  # Right
        
        current_speed = self.PLAYER_BASE_SPEED * self.player_speed_modifier
        self.player_vel = move_vec * current_speed

        if shift_held and not self.last_shift_state:
            # sound: card_cycle.wav
            available_cards = sorted(list(self.unlocked_card_indices))
            if len(available_cards) > 1:
                try:
                    current_selection_pos = available_cards.index(self.current_card_index)
                    next_selection_pos = (current_selection_pos + 1) % len(available_cards)
                    self.current_card_index = available_cards[next_selection_pos]
                except ValueError: # If current card was removed
                    self.current_card_index = available_cards[0]
        self.last_shift_state = shift_held

        if space_held and not self.last_space_state:
            card = self.all_cards[self.current_card_index]
            if self.energy >= card['cost'] and card['name'] not in self.active_cards:
                # sound: card_deploy.wav
                self.energy -= card['cost']
                self.active_cards[card['name']] = card['duration']
                self._apply_card_effect(card['name'])
                reward += 5.0
        self.last_space_state = space_held

        return reward

    def _update_player(self):
        self.player_pos += self.player_vel
        self.player_trail.append(self.player_pos.copy())
        if len(self.player_trail) > self.PLAYER_TRAIL_LENGTH:
            self.player_trail.pop(0)

        # Find current track boundaries at player's y-position
        player_y = self.player_pos[1]
        on_track = False
        for i in range(len(self.track_segments) - 1):
            s1 = self.track_segments[i]
            s2 = self.track_segments[i+1]
            if s1['y'] >= player_y >= s2['y']:
                on_track = True
                # Linear interpolation for precise boundaries
                ratio = (player_y - s2['y']) / max(1, (s1['y'] - s2['y']))
                center_x = s2['center_x'] + ratio * (s1['center_x'] - s2['center_x'])
                width = s2['width'] + ratio * (s1['width'] - s2['width'])
                left_bound = center_x - width / 2
                right_bound = center_x + width / 2
                
                if not (left_bound < self.player_pos[0] < right_bound):
                    # sound: collision_wall.wav
                    self.game_over = True
                self.player_pos[0] = np.clip(self.player_pos[0], left_bound, right_bound)
                break
        
        if not on_track and len(self.track_segments) > 1: # Handle cases at very top/bottom
            self.game_over = True

        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.screen_height)

    def _update_track(self):
        new_segments = []
        
        # Scroll existing segments
        for seg in self.track_segments:
            seg['y'] += self.TRACK_SCROLL_SPEED
            if seg['y'] < self.screen_height + 50:
                new_segments.append(seg)
        self.track_segments = new_segments

        # Generate new segments at the top
        while self.track_segments and self.track_segments[0]['y'] > -20:
            prev_seg = self.track_segments[0]
            new_center = prev_seg['center_x'] + self.np_random.uniform(-15, 15)
            new_center = np.clip(new_center, self.TRACK_WIDTH_MAX / 2, self.screen_width - self.TRACK_WIDTH_MAX / 2)
            new_width = prev_seg['width'] + self.np_random.uniform(-10, 10)
            new_width = np.clip(new_width, self.TRACK_WIDTH_MIN, self.TRACK_WIDTH_MAX)
            
            self.track_segments.insert(0, {
                'y': prev_seg['y'] - 20,
                'center_x': new_center,
                'width': new_width
            })

    def _update_particles(self):
        is_shielded = 'Shield' in self.active_cards
        for p in self.particles[:]:
            p['pos'][1] += self.particle_speed
            if p['pos'][1] > self.screen_height + p['size']:
                self.particles.remove(p)
                self.score += 1
                continue
            
            dist = np.linalg.norm(self.player_pos - p['pos'])
            if dist < self.PLAYER_SIZE + p['size']:
                if is_shielded:
                    # sound: shield_deflect.wav
                    self.particles.remove(p)
                    self.score += 5
                else:
                    # sound: collision_player.wav
                    self.game_over = True
                    return

    def _update_orbs(self):
        reward = 0
        is_magnet = 'Energy Magnet' in self.active_cards
        for orb in self.energy_orbs[:]:
            orb['pos'][1] += self.TRACK_SCROLL_SPEED / 2
            orb['angle'] = (orb['angle'] + 5) % 360

            if is_magnet:
                direction = self.player_pos - orb['pos']
                dist = np.linalg.norm(direction)
                if dist > 0 and dist < 150: # Magnet range
                    orb['pos'] += (direction / dist) * 3.0 # Magnet pull strength

            dist = np.linalg.norm(self.player_pos - orb['pos'])
            if dist < self.PLAYER_SIZE + orb['size']:
                # sound: collect_orb.wav
                self.energy_orbs.remove(orb)
                self.energy = min(100, self.energy + 15)
                self.score += 10
                reward += 1.0
                continue

            if orb['pos'][1] > self.screen_height + orb['size']:
                self.energy_orbs.remove(orb)
        return reward

    def _update_active_cards(self):
        for name in list(self.active_cards.keys()):
            self.active_cards[name] -= 1
            if self.active_cards[name] <= 0:
                del self.active_cards[name]
                if name == 'Speed Boost': self.player_speed_modifier = 1.0
                if name == 'Energy Magnet': pass # Effect stops when card expires

    def _spawn_entities(self):
        self.particle_spawn_timer -= 1
        if self.particle_spawn_timer <= 0:
            self._spawn_particle_wave()
            self.particle_spawn_timer = self.np_random.integers(70, 100) - self.steps // 100

        self.orb_spawn_timer -= 1
        if self.orb_spawn_timer <= 0:
            self._spawn_energy_orb()
            self.orb_spawn_timer = self.np_random.integers(50, 80)

    def _spawn_particle_wave(self):
        if not self.track_segments: return
        top_seg = self.track_segments[0]
        left = top_seg['center_x'] - top_seg['width'] / 2
        right = top_seg['center_x'] + top_seg['width'] / 2
        
        num_particles = self.np_random.integers(3, 6)
        pattern = self.np_random.choice(['line', 'v_shape', 'random'])
        
        for i in range(num_particles):
            size = self.np_random.uniform(5, 8)
            if pattern == 'line':
                x = left + (i + 1) * (top_seg['width'] / (num_particles + 1))
                y = -size
            elif pattern == 'v_shape':
                x = top_seg['center_x'] + (i - num_particles / 2) * 20
                y = -size - abs(i - num_particles / 2) * 15
            else: # random
                x = self.np_random.uniform(left + size, right - size)
                y = self.np_random.uniform(-40, -size)
                
            self.particles.append({'pos': np.array([x, y]), 'size': size})

    def _spawn_energy_orb(self):
        if not self.track_segments: return
        top_seg = self.track_segments[0]
        left = top_seg['center_x'] - top_seg['width'] / 2
        right = top_seg['center_x'] + top_seg['width'] / 2
        x = self.np_random.uniform(left + 10, right - 10)
        y = -10.0
        self.energy_orbs.append({'pos': np.array([x, y]), 'size': 8, 'angle': 0})

    def _update_difficulty(self):
        self.particle_speed = self.PARTICLE_BASE_SPEED + (self.steps / 200) * 0.05

    def _apply_card_effect(self, card_name):
        if card_name == 'Wave Clear':
            cleared_count = len(self.particles)
            self.particles.clear()
            self.score += 2 * cleared_count # Bonus for clearing
        elif card_name == 'Speed Boost':
            self.player_speed_modifier = 1.6
        # Shield and Magnet effects are handled in their respective update loops

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_track()
        self._render_orbs()
        self._render_particles()
        self._render_player()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_track(self):
        for i in range(len(self.track_segments) - 1):
            s1 = self.track_segments[i]
            s2 = self.track_segments[i+1]
            
            p1_left = (int(s1['center_x'] - s1['width'] / 2), int(s1['y']))
            p1_right = (int(s1['center_x'] + s1['width'] / 2), int(s1['y']))
            p2_left = (int(s2['center_x'] - s2['width'] / 2), int(s2['y']))
            p2_right = (int(s2['center_x'] + s2['width'] / 2), int(s2['y']))
            
            # Draw glowing lines
            for j in range(3, 0, -1):
                glow_alpha = self.COLOR_TRACK_GLOW[3] // (j * 2)
                pygame.draw.aaline(self.screen, (*self.COLOR_TRACK[:3], glow_alpha), p1_left, p2_left, j * 2)
                pygame.draw.aaline(self.screen, (*self.COLOR_TRACK[:3], glow_alpha), p1_right, p2_right, j * 2)

            pygame.draw.aaline(self.screen, self.COLOR_TRACK, p1_left, p2_left)
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, p1_right, p2_right)

    def _render_player(self):
        # Trail
        trail_color = self.COLOR_SPEED_BOOST_TRAIL if 'Speed Boost' in self.active_cards else self.COLOR_PLAYER
        for i, pos in enumerate(self.player_trail):
            alpha = int(255 * (i / self.PLAYER_TRAIL_LENGTH))
            radius = int(self.PLAYER_SIZE * 0.5 * (i / self.PLAYER_TRAIL_LENGTH))
            if radius > 1:
                self._draw_glowing_circle(self.screen, (*trail_color, alpha//2), pos, radius, (*trail_color, alpha//4))

        # Shield
        if 'Shield' in self.active_cards:
            duration = self.active_cards['Shield']
            pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
            radius = self.PLAYER_SIZE * 1.8 + pulse * 4
            # Fade out shield effect
            alpha = min(self.COLOR_SHIELD[3], int(255 * (duration / 30)))
            pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos[0]), int(self.player_pos[1]), int(radius), (*self.COLOR_SHIELD[:3], alpha//2))
            pygame.gfxdraw.aacircle(self.screen, int(self.player_pos[0]), int(self.player_pos[1]), int(radius), (*self.COLOR_SHIELD[:3], alpha))

        # Player vehicle
        self._draw_glowing_circle(self.screen, self.COLOR_PLAYER, self.player_pos, self.PLAYER_SIZE, self.COLOR_PLAYER_GLOW)

    def _render_particles(self):
        for p in self.particles:
            pulse = (math.sin(self.steps * 0.1 + p['pos'][0]) + 1) / 2
            glow_radius = p['size'] + pulse * 4
            self._draw_glowing_circle(self.screen, self.COLOR_PARTICLE, p['pos'], int(p['size']), (*self.COLOR_PARTICLE_GLOW[:3], int(self.COLOR_PARTICLE_GLOW[3] * pulse)))

    def _render_orbs(self):
        for orb in self.energy_orbs:
            self._draw_glowing_circle(self.screen, self.COLOR_ORB, orb['pos'], orb['size'], self.COLOR_ORB_GLOW)
            # Spinning effect
            for i in range(2):
                angle = math.radians(orb['angle'] + i * 180)
                start = orb['pos'] + np.array([math.cos(angle), math.sin(angle)]) * orb['size'] * 0.8
                end = orb['pos'] - np.array([math.cos(angle), math.sin(angle)]) * orb['size'] * 0.8
                pygame.draw.aaline(self.screen, (255,255,255,100), start, end)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Energy Bar
        energy_rect_bg = pygame.Rect(self.screen_width - 160, 15, 150, 15)
        pygame.draw.rect(self.screen, self.COLOR_ENERGY_BAR_BG, energy_rect_bg, border_radius=3)
        energy_width = int(148 * (self.energy / 100))
        energy_rect = pygame.Rect(self.screen_width - 159, 16, energy_width, 13)
        pygame.draw.rect(self.screen, self.COLOR_ENERGY_BAR, energy_rect, border_radius=3)
        energy_text = self.font_ui.render("ENERGY", True, self.COLOR_UI_TEXT)
        self.screen.blit(energy_text, (self.screen_width - 230, 12))

        # Current Card
        card = self.all_cards[self.current_card_index]
        card_text_str = f"CARD: {card['name']} (Cost: {card['cost']})"
        card_color = self.COLOR_UI_TEXT
        if self.energy < card['cost'] or card['name'] in self.active_cards:
            card_color = (120, 120, 120) # Grey out if unusable
        card_text = self.font_card.render(card_text_str, True, card_color)
        text_rect = card_text.get_rect(center=(self.screen_width / 2, self.screen_height - 30))
        self.screen.blit(card_text, text_rect)

        # Speed
        speed_val = self.particle_speed + self.TRACK_SCROLL_SPEED
        speed_text = self.font_ui.render(f"SPEED: {speed_val:.1f}", True, self.COLOR_UI_TEXT)
        speed_rect = speed_text.get_rect(bottomright=(self.screen_width - 10, self.screen_height - 10))
        self.screen.blit(speed_text, speed_rect)

    def _draw_glowing_circle(self, surface, color, center, radius, glow_color):
        center_int = (int(center[0]), int(center[1]))
        # Draw multiple layers for a soft glow
        for i in range(int(radius), 0, -2):
            alpha = glow_color[3] * (i / radius)**2
            pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], int(radius + (radius - i) * 0.4), (*glow_color[:3], int(alpha)))
        
        pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], int(radius), color)
        pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], int(radius), color)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "energy": self.energy}

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not run in a headless environment
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11"
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        # Setup for manual play
        screen = pygame.display.set_mode((640, 400))
        pygame.display.set_caption("Quantum Tunnel Racer")
        clock = pygame.time.Clock()
        
        terminated = False
        truncated = False
        total_reward = 0
        
        # Action state
        movement_action = 0 # 0: none
        space_action = 0    # 0: released
        shift_action = 0    # 0: released
        
        print("\n--- Manual Control ---")
        print(GameEnv.user_guide)
        print("Q: Quit")
        
        while not terminated and not truncated:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    terminated = True

            # Get key presses for this frame
            keys = pygame.key.get_pressed()
            
            # Movement
            if keys[pygame.K_UP]: movement_action = 1
            elif keys[pygame.K_DOWN]: movement_action = 2
            elif keys[pygame.K_LEFT]: movement_action = 3
            elif keys[pygame.K_RIGHT]: movement_action = 4
            else: movement_action = 0
                
            # Abilities
            space_action = 1 if keys[pygame.K_SPACE] else 0
            shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            # Construct the action tuple
            action = [movement_action, space_action, shift_action]
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Render the observation from the environment to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(30) # Run at 30 FPS

            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

        env.close()
    except pygame.error as e:
        print(f"Pygame error (likely due to headless environment): {e}")
        print("Manual execution is not possible in a headless environment.")
        # Create a dummy env to allow for basic checks
        env = GameEnv()
        env.reset()
        env.step(env.action_space.sample())
        env.close()
        print("Dummy environment created and tested successfully in headless mode.")