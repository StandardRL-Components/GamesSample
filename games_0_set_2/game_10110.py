import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:56:39.562645
# Source Brief: brief_00110.md
# Brief Index: 110
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player protects a central Grimoire from shadowy
    intruders by placing and upgrading magnetic runes. The game prioritizes visual
    polish and a satisfying strategic experience.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend the central Grimoire from shadowy intruders by placing and upgrading magnetic runes. "
        "Strategically position your defenses to repel enemies before they reach the core."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the selected rune. "
        "Press space to place a new rune. Press shift to upgrade the selected rune and cycle selection."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    MAX_STEPS = 1000
    MAX_RUNES = 3
    
    # Colors
    COLOR_BG = (15, 10, 30)
    COLOR_GRIMOIRE_MAIN = (255, 0, 200)
    COLOR_GRIMOIRE_GLOW = (80, 0, 60)
    COLOR_INTRUDER = (70, 70, 90)
    COLOR_INTRUDER_GLOW = (150, 150, 180)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_SELECTION = (255, 255, 255)

    RUNE_PROPS = {
        1: {'color': (0, 150, 255), 'force': 4000, 'radius': 12, 'glow_radius': 20, 'max_field_dist': 150},
        2: {'color': (0, 255, 150), 'force': 8000, 'radius': 14, 'glow_radius': 25, 'max_field_dist': 200},
        3: {'color': (255, 50, 50), 'force': 16000, 'radius': 16, 'glow_radius': 30, 'max_field_dist': 250},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 50)
        
        # Game state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.grimoire_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.grimoire_radius = 25
        self.runes = []
        self.intruders = []
        self.particles = []
        self.selected_rune_idx = -1
        self.last_space_held = False
        self.last_shift_held = False
        self.max_intruders = 1
        self.current_intruder_speed = 1.0
        
        # self.validate_implementation() # Removed for submission
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.runes = []
        self.intruders = []
        self.particles = []
        self.selected_rune_idx = -1
        self.last_space_held = False
        self.last_shift_held = False
        self.max_intruders = 1
        self.current_intruder_speed = 1.0
        
        self._spawn_intruder()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.1  # Survival reward

        # --- Handle Actions ---
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held

        if self.selected_rune_idx != -1 and len(self.runes) > 0:
            self._handle_movement(movement)
        
        if space_pressed:
            self._handle_rune_placement()

        if shift_pressed:
            self._handle_upgrade_and_cycle()

        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # --- Update Game State ---
        repelled_count = self._update_intruders()
        self._update_particles()
        
        # --- Update Difficulty ---
        if self.steps > 0 and self.steps % 200 == 0:
            self.current_intruder_speed = min(3.0, self.current_intruder_speed + 0.05)
        if self.steps > 0 and self.steps % 500 == 0:
            self.max_intruders = min(5, self.max_intruders + 1)
        
        while len(self.intruders) < self.max_intruders:
            self._spawn_intruder()
            
        # --- Calculate Reward & Termination ---
        reward += repelled_count * 1.0
        self.score += repelled_count
        
        terminated = self._check_termination()
        
        if terminated and not self.game_over: # Game just ended this step
            if self.steps >= self.MAX_STEPS:
                reward += 100.0 # Victory bonus
            else:
                reward = -100.0 # Loss penalty
            self.game_over = True
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    # --- Game Logic Helpers ---

    def _handle_movement(self, movement):
        rune = self.runes[self.selected_rune_idx]
        move_speed = 5
        if movement == 1: rune['pos'].y -= move_speed
        elif movement == 2: rune['pos'].y += move_speed
        elif movement == 3: rune['pos'].x -= move_speed
        elif movement == 4: rune['pos'].x += move_speed

        rune['pos'].x = np.clip(rune['pos'].x, 0, self.SCREEN_WIDTH)
        rune['pos'].y = np.clip(rune['pos'].y, 0, self.SCREEN_HEIGHT)

    def _handle_rune_placement(self):
        if len(self.runes) < self.MAX_RUNES:
            # Place at a random location not too close to the center
            while True:
                pos = pygame.Vector2(random.uniform(50, self.SCREEN_WIDTH - 50),
                                     random.uniform(50, self.SCREEN_HEIGHT - 50))
                if pos.distance_to(self.grimoire_pos) > self.grimoire_radius + 50:
                    break
            
            self.runes.append({'pos': pos, 'level': 1})
            self.selected_rune_idx = len(self.runes) - 1
            # sfx: rune placed
            self._spawn_particles(pos, self.RUNE_PROPS[1]['color'], 30, 3)

    def _handle_upgrade_and_cycle(self):
        if not self.runes:
            return

        # Upgrade
        if self.selected_rune_idx != -1:
            rune = self.runes[self.selected_rune_idx]
            if rune['level'] < 3:
                rune['level'] += 1
                # sfx: rune upgrade
                self._spawn_particles(rune['pos'], self.RUNE_PROPS[rune['level']]['color'], 50, 5)

        # Cycle selection
        if len(self.runes) > 0:
            self.selected_rune_idx = (self.selected_rune_idx + 1) % len(self.runes)


    def _update_intruders(self):
        repelled_count = 0
        for intruder in self.intruders[:]:
            # Attraction to Grimoire
            dir_to_grimoire = (self.grimoire_pos - intruder['pos']).normalize()
            intruder['vel'] = dir_to_grimoire * self.current_intruder_speed
            
            # Repulsion from Runes
            total_repulsion = pygame.Vector2(0, 0)
            for rune in self.runes:
                dist_vec = intruder['pos'] - rune['pos']
                dist_sq = dist_vec.length_squared()
                if dist_sq > 1: # Avoid division by zero
                    force_magnitude = self.RUNE_PROPS[rune['level']]['force'] / dist_sq
                    total_repulsion += dist_vec.normalize() * force_magnitude
            
            intruder['vel'] += total_repulsion
            
            # Clamp speed
            if intruder['vel'].length() > self.current_intruder_speed * 2.5:
                intruder['vel'].scale_to_length(self.current_intruder_speed * 2.5)
            
            intruder['pos'] += intruder['vel']

            # Check for termination/repel
            if intruder['pos'].distance_to(self.grimoire_pos) < self.grimoire_radius:
                self.game_over = True
                # sfx: game over
                self._spawn_particles(intruder['pos'], (255,0,0), 100, 10)
            
            if not self.screen.get_rect().collidepoint(intruder['pos']):
                self.intruders.remove(intruder)
                repelled_count += 1
                # sfx: intruder repelled
        return repelled_count

    def _spawn_intruder(self):
        side = random.choice(['top', 'bottom', 'left', 'right'])
        if side == 'top':
            pos = pygame.Vector2(random.uniform(0, self.SCREEN_WIDTH), -20)
        elif side == 'bottom':
            pos = pygame.Vector2(random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 20)
        elif side == 'left':
            pos = pygame.Vector2(-20, random.uniform(0, self.SCREEN_HEIGHT))
        else: # right
            pos = pygame.Vector2(self.SCREEN_WIDTH + 20, random.uniform(0, self.SCREEN_HEIGHT))
        
        self.intruders.append({'pos': pos, 'vel': pygame.Vector2(0,0), 'anim_offset': random.uniform(0, 2 * math.pi)})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _spawn_particles(self, pos, color, count, speed_scale):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 1.5) * speed_scale
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': random.randint(15, 30),
                'color': color,
                'size': random.uniform(1, 4)
            })

    def _check_termination(self):
        return self.game_over or self.steps >= self.MAX_STEPS

    # --- Rendering Helpers ---
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grimoire_glow()
        self._render_field_lines()
        self._render_runes()
        self._render_grimoire()
        self._render_intruders()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grimoire(self):
        pos = (int(self.grimoire_pos.x), int(self.grimoire_pos.y))
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.grimoire_radius, self.COLOR_GRIMOIRE_MAIN)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.grimoire_radius, self.COLOR_GRIMOIRE_MAIN)
        
        # Simple sigil
        for i in range(5):
            angle = self.steps * 0.01 + (i * 2 * math.pi / 5)
            start_angle = angle + math.pi / 5
            end_angle = angle - math.pi / 5
            p1 = self.grimoire_pos + pygame.Vector2(math.cos(start_angle), math.sin(start_angle)) * self.grimoire_radius * 0.8
            p2 = self.grimoire_pos + pygame.Vector2(math.cos(end_angle), math.sin(end_angle)) * self.grimoire_radius * 0.8
            pygame.draw.aaline(self.screen, self.COLOR_BG, p1, p2, 2)


    def _render_grimoire_glow(self):
        # Pulsating glow effect
        pulse = (math.sin(self.steps * 0.05) + 1) / 2
        max_glow_radius = self.grimoire_radius * 2.5
        current_glow_radius = int(self.grimoire_radius + pulse * (max_glow_radius - self.grimoire_radius))
        
        temp_surf = pygame.Surface((current_glow_radius * 2, current_glow_radius * 2), pygame.SRCALPHA)
        glow_color_alpha = self.COLOR_GRIMOIRE_GLOW + (int(100 - pulse * 50),)
        pygame.draw.circle(temp_surf, glow_color_alpha, (current_glow_radius, current_glow_radius), current_glow_radius)
        self.screen.blit(temp_surf, (self.grimoire_pos.x - current_glow_radius, self.grimoire_pos.y - current_glow_radius), special_flags=pygame.BLEND_RGBA_ADD)


    def _render_runes(self):
        for i, rune in enumerate(self.runes):
            props = self.RUNE_PROPS[rune['level']]
            pos = (int(rune['pos'].x), int(rune['pos'].y))
            
            # Glow
            pulse = (math.sin(self.steps * 0.1 + i) + 1) / 2
            glow_radius = int(props['glow_radius'] * (0.8 + pulse * 0.4))
            glow_color = tuple(min(255, int(c * 0.4)) for c in props['color'])
            
            temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, glow_color + (100,), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(temp_surf, (pos[0] - glow_radius, pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
            
            # Core
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], props['radius'], props['color'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], props['radius'], props['color'])

            # Level indicator
            level_text = self.font_small.render(str(rune['level']), True, self.COLOR_UI_TEXT)
            self.screen.blit(level_text, (pos[0] - level_text.get_width() / 2, pos[1] - props['radius'] - 20))

            # Selection indicator
            if i == self.selected_rune_idx:
                sel_pulse = (math.sin(self.steps * 0.2) + 1) / 2
                sel_radius = int(props['radius'] + 5 + sel_pulse * 3)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], sel_radius, self.COLOR_SELECTION)


    def _render_intruders(self):
        for intruder in self.intruders:
            pos = intruder['pos']
            anim_sin = math.sin(self.steps * 0.15 + intruder['anim_offset'])
            size = 10 + anim_sin * 2
            
            # Directional triangle
            angle = math.atan2(intruder['vel'].y, intruder['vel'].x)
            points = [
                pos + pygame.Vector2(size, 0).rotate_rad(angle),
                pos + pygame.Vector2(-size * 0.5, -size * 0.8).rotate_rad(angle),
                pos + pygame.Vector2(-size * 0.5, size * 0.8).rotate_rad(angle)
            ]
            
            int_points = [(int(p.x), int(p.y)) for p in points]
            
            pygame.gfxdraw.filled_trigon(self.screen, int_points[0][0], int_points[0][1], int_points[1][0], int_points[1][1], int_points[2][0], int_points[2][1], self.COLOR_INTRUDER)
            pygame.gfxdraw.aatrigon(self.screen, int_points[0][0], int_points[0][1], int_points[1][0], int_points[1][1], int_points[2][0], int_points[2][1], self.COLOR_INTRUDER_GLOW)


    def _render_field_lines(self):
        for intruder in self.intruders:
            for rune in self.runes:
                dist_vec = intruder['pos'] - rune['pos']
                dist = dist_vec.length()
                props = self.RUNE_PROPS[rune['level']]

                if dist < props['max_field_dist']:
                    alpha = int(max(0, min(255, (1 - (dist / props['max_field_dist']))**2 * 100)))
                    if alpha > 5:
                        pygame.draw.aaline(self.screen, props['color'] + (alpha,), intruder['pos'], rune['pos'])

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30.0))
            color_with_alpha = p['color'] + (alpha,)
            pygame.draw.circle(self.screen, color_with_alpha, p['pos'], p['size'])

    def _render_ui(self):
        # Timer
        timer_text = self.font_small.render(f"Time: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (10, 10))

        # Score
        score_text = self.font_small.render(f"Repelled: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, self.SCREEN_HEIGHT - 30))
        
        # Game Over/Victory Message
        if self.game_over:
            if self.steps >= self.MAX_STEPS:
                msg = "GRIMOIRE SECURED"
                color = (0, 255, 150)
            else:
                msg = "GRIMOIRE LOST"
                color = (255, 50, 50)
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "runes_active": len(self.runes),
            "intruders_active": len(self.intruders)
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To run, you need to unset the dummy video driver
    # e.g., by commenting out the os.environ.setdefault line
    # or running: unset SDL_VIDEODRIVER
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for manual play
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Forbidden Grimoire Defense")
    clock = pygame.time.Clock()

    movement, space, shift = 0, 0, 0
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            
            # Key presses for manual control
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: space = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE: space = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift = 0

        keys = pygame.key.get_pressed()
        movement = 0 # No movement
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Run at 30 FPS

    env.close()