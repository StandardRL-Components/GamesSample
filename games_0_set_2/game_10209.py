import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T09:59:39.932265
# Source Brief: brief_00209.md
# Brief Index: 209
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np


# Helper function for drawing anti-aliased circles, crucial for visual quality
def draw_aacircle(surface, color, center, radius):
    """Draws an anti-aliased circle with a filled center."""
    x, y = int(center[0]), int(center[1])
    rad = int(radius)
    if rad <= 0: return

    # Draw the anti-aliased outline
    pygame.gfxdraw.aacircle(surface, x, y, rad, color)
    # Draw the filled circle
    pygame.gfxdraw.filled_circle(surface, x, y, rad, color)

class GameEnv(gym.Env):
    """
    A Gymnasium environment where a fungal spore infiltrates a plant's vascular system.
    The agent must navigate past immune cells to inject enzymes into weak points.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Infiltrate a plant's vascular system as a fungal spore. Navigate past immune cells "
        "and inject enzymes into weak points to succeed."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move. Press space to inject enzymes into targets and "
        "shift to flip the vertical controls."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- CRITICAL: Gymnasium Space Definitions ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Game & Visual Configuration ---
        self.W, self.H = 640, 400
        self.FPS = 30 # Assumed FPS for smooth interpolation
        self.MAX_STEPS = 2000

        # --- Color Palette ---
        self.COLOR_BG = (10, 30, 25)
        self.COLOR_WALLS = (20, 80, 60)
        self.COLOR_PLAYER = (255, 50, 50)
        self.COLOR_PLAYER_GLOW = (255, 50, 50)
        self.COLOR_ENEMY = (220, 220, 255)
        self.COLOR_ENEMY_GLOW = (220, 220, 255)
        self.COLOR_WEAK_POINT = (255, 255, 0)
        self.COLOR_ENZYME = (150, 0, 255)
        self.COLOR_UI_TEXT = (200, 200, 200)
        self.COLOR_GRAVITY_UP = (50, 150, 255)
        self.COLOR_GRAVITY_DOWN = (255, 150, 50)

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # --- Initialize state variables (will be properly set in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.array([0.0, 0.0])
        self.player_radius = 0
        self.player_speed = 0
        self.gravity_flipped = False
        self.last_shift_state = 0
        self.last_space_state = 0
        self.enzyme_cooldown = 0
        self.weak_points = []
        self.enemies = []
        self.effects = []
        self.enemy_paths = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Core Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False

        # --- Player State ---
        # FIX: Moved player start position to avoid immediate collision with an enemy path.
        self.player_pos = np.array([100.0, 150.0])
        self.player_radius = 12
        self.player_speed = 3.0

        # --- Action Handling State ---
        self.gravity_flipped = False
        self.last_shift_state = 0
        self.last_space_state = 0
        self.enzyme_cooldown = 0
        self.enzyme_cooldown_max = 30 # 1 sec @ 30 FPS

        # --- World State ---
        self.weak_points = [
            {'pos': np.array([580.0, 80.0]), 'radius': 15, 'active': True},
            {'pos': np.array([580.0, 320.0]), 'radius': 15, 'active': True},
            {'pos': np.array([320.0, 200.0]), 'radius': 15, 'active': True},
        ]

        self.enemy_paths = [
            [np.array([200, 50]), np.array([200, 350])],
            [np.array([450, 350]), np.array([450, 50])],
            [np.array([50, 200]), np.array([250, 200]), np.array([400, 350]), np.array([550, 200]), np.array([400, 50]), np.array([250, 200])]
        ]

        self.enemies = []
        base_enemy_speed = 1.5
        for i, path in enumerate(self.enemy_paths):
            self.enemies.append({
                'pos': path[0].copy().astype(float),
                'radius': 10,
                'path': path,
                'path_index': 1,
                'speed': base_enemy_speed + i * 0.2,
                'direction': 1
            })

        self.effects = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01  # Small cost for existing

        # --- Update Game Logic ---
        reward_event = self._update_player(action)
        self._update_enemies()
        self._update_effects()
        reward += reward_event
        self.score += reward_event

        # --- Check Termination Conditions ---
        terminated = False
        truncated = False
        if self._check_collisions():
            reward -= 100
            terminated = True
        elif self._check_win_condition():
            reward += 100
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True # Use truncated for time limit

        self.game_over = terminated or truncated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_player(self, action):
        movement, space_held, shift_held = action[0], action[1], action[2]

        # --- Handle Movement ---
        move_vector = np.array([0.0, 0.0])
        up_action = 1
        down_action = 2

        # Gravity flip swaps the meaning of up/down actions
        if self.gravity_flipped:
            up_action, down_action = down_action, up_action

        if movement == up_action: move_vector[1] = -1
        elif movement == down_action: move_vector[1] = 1
        elif movement == 3: move_vector[0] = -1
        elif movement == 4: move_vector[0] = 1

        if np.linalg.norm(move_vector) > 0:
            move_vector = move_vector / np.linalg.norm(move_vector)
        self.player_pos += move_vector * self.player_speed

        # --- Boundary Checks ---
        self.player_pos[0] = np.clip(self.player_pos[0], self.player_radius, self.W - self.player_radius)
        self.player_pos[1] = np.clip(self.player_pos[1], self.player_radius, self.H - self.player_radius)

        # --- Handle Actions (on-press logic) ---
        reward = 0

        # Gravity Flip (Shift)
        if shift_held and not self.last_shift_state:
            self.gravity_flipped = not self.gravity_flipped
        self.last_shift_state = shift_held

        # Enzyme Injection (Space)
        if self.enzyme_cooldown > 0:
            self.enzyme_cooldown -= 1

        if space_held and not self.last_space_state and self.enzyme_cooldown == 0:
            for wp in self.weak_points:
                if wp['active']:
                    dist = np.linalg.norm(self.player_pos - wp['pos'])
                    if dist < self.player_radius + wp['radius'] + 5: # 5px grace
                        wp['active'] = False
                        reward += 50 # Increased reward for hitting a target
                        self.enzyme_cooldown = self.enzyme_cooldown_max
                        self._create_enzyme_effect(wp['pos'])
                        break # Inject into one at a time
        self.last_space_state = space_held

        return reward

    def _update_enemies(self):
        # Difficulty scaling
        speed_multiplier = 1.0 + (self.steps // 500) * 0.05

        for enemy in self.enemies:
            target_pos = enemy['path'][enemy['path_index']]
            direction_vec = target_pos - enemy['pos']
            dist = np.linalg.norm(direction_vec)

            if dist < enemy['speed'] * speed_multiplier:
                enemy['pos'] = target_pos.copy()
                if len(enemy['path']) > 1: # Avoid issues with single-point paths
                    enemy['path_index'] += enemy['direction']
                    if not (0 <= enemy['path_index'] < len(enemy['path'])):
                        enemy['direction'] *= -1
                        enemy['path_index'] += 2 * enemy['direction']
            else:
                move_vec = (direction_vec / dist) * enemy['speed'] * speed_multiplier
                enemy['pos'] += move_vec

    def _update_effects(self):
        for effect in self.effects[:]:
            effect['timer'] -= 1
            if effect['timer'] <= 0:
                self.effects.remove(effect)
                continue

            if effect['type'] == 'enzyme_spread':
                effect['radius'] += 0.5
                effect['alpha'] = max(0, 255 * (effect['timer'] / effect['duration']))

            if effect['type'] == 'particle':
                effect['pos'] += effect['vel']
                effect['vel'] *= 0.98 # Friction
                effect['alpha'] = max(0, 255 * (effect['timer'] / effect['duration']))

    def _create_enzyme_effect(self, pos):
        # Main spreading effect
        self.effects.append({
            'type': 'enzyme_spread', 'pos': pos.copy(), 'radius': 5,
            'timer': 60, 'duration': 60, 'alpha': 255
        })
        # Particle burst
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.effects.append({
                'type': 'particle', 'pos': pos.copy(), 'vel': vel,
                'radius': random.uniform(1, 3), 'timer': 40, 'duration': 40, 'alpha': 255
            })

    def _check_collisions(self):
        for enemy in self.enemies:
            dist = np.linalg.norm(self.player_pos - enemy['pos'])
            if dist < self.player_radius + enemy['radius']:
                return True
        return False

    def _check_win_condition(self):
        return not any(wp['active'] for wp in self.weak_points)

    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_background()
        self._render_effects()
        self._render_weak_points()
        self._render_enemies()
        self._render_player()

        # Render UI overlay
        self._render_ui()

        if self.game_over:
            self._render_game_over()

        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for path in self.enemy_paths:
            if len(path) > 1:
                pygame.draw.lines(self.screen, self.COLOR_WALLS, False, [tuple(p) for p in path], 10)

    def _render_effects(self):
        for effect in self.effects:
            pos = (int(effect['pos'][0]), int(effect['pos'][1]))
            if effect['type'] == 'enzyme_spread':
                radius = int(effect['radius'])
                if radius <= 0: continue
                s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                color = (*self.COLOR_ENZYME, int(effect['alpha']))
                pygame.draw.circle(s, color, (radius, radius), radius)
                self.screen.blit(s, (pos[0] - radius, pos[1] - radius))
            elif effect['type'] == 'particle':
                color = (*self.COLOR_ENZYME, int(effect['alpha']))
                draw_aacircle(self.screen, color, effect['pos'], effect['radius'])

    def _render_weak_points(self):
        for wp in self.weak_points:
            if wp['active']:
                draw_aacircle(self.screen, self.COLOR_WEAK_POINT, wp['pos'], wp['radius'])

    def _render_enemies(self):
        for enemy in self.enemies:
            # Pulsating glow effect
            glow_radius = enemy['radius'] + 4 + 2 * math.sin(self.steps * 0.1)
            s = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
            glow_alpha = 100 + 40 * math.sin(self.steps * 0.1)
            color = (*self.COLOR_ENEMY_GLOW, int(glow_alpha))
            pygame.draw.circle(s, color, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(s, (enemy['pos'][0] - glow_radius, enemy['pos'][1] - glow_radius))

            # Main body
            draw_aacircle(self.screen, self.COLOR_ENEMY, enemy['pos'], enemy['radius'])

    def _render_player(self):
        # Glow effect
        glow_radius = self.player_radius + 6
        s = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        color = (*self.COLOR_PLAYER_GLOW, 100)
        pygame.draw.circle(s, color, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (self.player_pos[0] - glow_radius, self.player_pos[1] - glow_radius))

        # Main body
        draw_aacircle(self.screen, self.COLOR_PLAYER, self.player_pos, self.player_radius)

    def _render_ui(self):
        # Weak points remaining
        remaining_wp = sum(1 for wp in self.weak_points if wp['active'])
        text_surface = self.font_ui.render(f"TARGETS: {remaining_wp}", True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surface, (10, 10))

        # Gravity indicator
        grav_text = "GRAV: REVERSED" if self.gravity_flipped else "GRAV: NORMAL"
        grav_color = self.COLOR_GRAVITY_UP if self.gravity_flipped else self.COLOR_GRAVITY_DOWN
        text_surface = self.font_ui.render(grav_text, True, grav_color)
        self.screen.blit(text_surface, (self.W - text_surface.get_width() - 10, 10))

    def _render_game_over(self):
        overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))

        win = self._check_win_condition()
        text = "SYSTEM WEAKENED" if win else "SPORE NEUTRALIZED"
        color = (100, 255, 100) if win else (255, 100, 100)

        text_surface = self.font_game_over.render(text, True, color)
        text_rect = text_surface.get_rect(center=(self.W / 2, self.H / 2))

        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "remaining_weak_points": sum(1 for wp in self.weak_points if wp['active']),
            "player_pos": self.player_pos.tolist(),
            "gravity_flipped": self.gravity_flipped,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires pygame to be installed and will open a window.
    # To run headless, this block should not be executed.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Vascular Invader")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0.0
    game_over_state = False

    while running:
        # --- Action Mapping for Manual Play ---
        movement = 0 # none
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if not game_over_state:
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4

            if keys[pygame.K_SPACE]: space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0.0
                game_over_state = False

        # --- Step the Environment ---
        if not game_over_state:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # --- Render to Screen ---
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated or truncated:
                print(f"Episode finished. Total reward: {total_reward:.2f}, Info: {info}")
                game_over_state = True
        
        clock.tick(env.FPS)

    env.close()