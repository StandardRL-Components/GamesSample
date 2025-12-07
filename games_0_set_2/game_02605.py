
# Generated: 2025-08-28T05:22:37.200100
# Source Brief: brief_02605.md
# Brief Index: 2605

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move the cursor. Press space to squash bugs. Avoid the red traps!"
    )

    game_description = (
        "Squash creepy crawlies while avoiding deadly traps in a procedurally generated isometric horror environment. Squash 20 bugs to win, but trigger 3 traps and you lose."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 2000
        self.WIN_SCORE = 20
        self.LOSE_CONDITION = 3
        
        # --- Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)

        # --- Colors ---
        self.COLOR_BG = (20, 20, 25)
        self.COLOR_TILE_1 = (35, 35, 40)
        self.COLOR_TILE_2 = (40, 40, 45)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_CURSOR_WARN = (255, 165, 0)
        self.COLOR_CURSOR_DANGER = (255, 60, 60)
        self.COLOR_BUG_OUTER = (50, 150, 50)
        self.COLOR_BUG_INNER = (100, 255, 100)
        self.COLOR_TRAP_BASE = (100, 20, 20)
        self.COLOR_TRAP_GLOW = (200, 40, 40)
        self.COLOR_TEXT = (220, 220, 220)

        # --- Game State (initialized in reset) ---
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.bugs_squashed = 0
        self.traps_triggered = 0
        self.game_over = False
        self.win = False

        self.cursor_pos = None
        self.cursor_speed = 5.0
        self.trap_proximity_threshold = 50

        self.bug_speed = 1.0
        self.bugs = []
        self.traps = []
        self.particles = []
        self.squash_effects = []
        self.squash_cooldown = 0
        self.squash_radius = 30
        
        self.tile_map = []

        # Initialize state and validate
        self.reset()
        self.validate_implementation()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.bugs_squashed = 0
        self.traps_triggered = 0
        self.game_over = False
        self.win = False
        
        self.bug_speed = 1.0
        self.cursor_pos = np.array([self.WIDTH / 2.0, self.HEIGHT / 2.0])
        
        self.bugs.clear()
        self.traps.clear()
        self.particles.clear()
        self.squash_effects.clear()
        self.squash_cooldown = 0

        # Generate a static tile map for visual consistency
        if not self.tile_map:
            self._generate_tile_map()

        for _ in range(10):
            self._spawn_bug()
        for _ in range(5):
            self._spawn_trap()
            
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        if self.squash_cooldown > 0:
            self.squash_cooldown -= 1

        # --- Continuous Rewards Pre-Move ---
        dist_bug_before = self._get_min_dist_to_entity(self.bugs)
        dist_trap_before = self._get_min_dist_to_entity(self.traps)

        # --- Handle Action ---
        movement, space_pressed, _ = action[0], action[1] == 1, action[2] == 1
        self._update_cursor(movement)

        # --- Continuous Rewards Post-Move ---
        dist_bug_after = self._get_min_dist_to_entity(self.bugs)
        dist_trap_after = self._get_min_dist_to_entity(self.traps)
        
        if dist_bug_after < dist_bug_before:
            reward += 0.5
        if dist_trap_after < dist_trap_before:
            reward -= 0.1

        # --- Update Game Logic ---
        self._update_bugs()
        
        # Check for trap triggers
        triggered_trap = self._check_trap_collision()
        if triggered_trap:
            self.traps_triggered += 1
            self.score -= 20
            reward -= 20
            self.traps.remove(triggered_trap)
            self._create_particles(triggered_trap['pos'], self.COLOR_TRAP_GLOW, 50, 5)
            # sfx: trap spring
            self._spawn_trap()

        # Check for squash action
        if space_pressed and self.squash_cooldown == 0:
            self.squash_cooldown = 15  # Cooldown of 0.5s at 30fps
            self._create_squash_effect()
            # sfx: squash sound
            
            squashed_this_frame = 0
            for bug in self.bugs[:]:
                dist = np.linalg.norm(self.cursor_pos - bug['pos'])
                if dist < self.squash_radius:
                    self.bugs_squashed += 1
                    squashed_this_frame += 1
                    self.score += 10
                    reward += 10
                    self.bugs.remove(bug)
                    self._create_particles(bug['pos'], self.COLOR_BUG_INNER, 30, 3)
                    self._spawn_bug()
        
        self._update_effects()
        self._update_difficulty()

        # --- Check Termination ---
        terminated = False
        if self.bugs_squashed >= self.WIN_SCORE:
            self.score += 100
            reward += 100
            terminated = True
            self.win = True
            self.game_over = True
        elif self.traps_triggered >= self.LOSE_CONDITION:
            self.score -= 100
            reward -= 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    # --- Update Helpers ---
    def _update_cursor(self, movement):
        if movement == 1: self.cursor_pos[1] -= self.cursor_speed
        elif movement == 2: self.cursor_pos[1] += self.cursor_speed
        elif movement == 3: self.cursor_pos[0] -= self.cursor_speed
        elif movement == 4: self.cursor_pos[0] += self.cursor_speed
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)

    def _update_bugs(self):
        for bug in self.bugs:
            dist_to_target = np.linalg.norm(bug['pos'] - bug['target'])
            if dist_to_target < self.bug_speed * 2:
                bug['target'] = self._get_safe_spawn_pos(min_dist=0)
            
            direction = (bug['target'] - bug['pos']) / (dist_to_target + 1e-6)
            bug['pos'] += direction * self.bug_speed
            bug['anim_offset'] += 0.1

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 500 == 0:
            self.bug_speed = min(3.0, self.bug_speed + 0.05)

    def _update_effects(self):
        self.particles[:] = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        
        self.squash_effects[:] = [e for e in self.squash_effects if e['life'] > 0]
        for e in self.squash_effects:
            e['life'] -= 1

    # --- Collision and Spawning ---
    def _check_trap_collision(self):
        for trap in self.traps:
            dist = np.linalg.norm(self.cursor_pos - trap['pos'])
            if dist < trap['radius']:
                return trap
        return None

    def _get_safe_spawn_pos(self, min_dist=100, padding=20):
        for _ in range(100): # Max 100 attempts
            pos = np.array([
                self.np_random.uniform(padding, self.WIDTH - padding),
                self.np_random.uniform(padding, self.HEIGHT - padding)
            ])
            if np.linalg.norm(pos - self.cursor_pos) < min_dist:
                continue
            if any(np.linalg.norm(pos - e['pos']) < 30 for e in self.bugs + self.traps):
                continue
            return pos
        return np.array([self.WIDTH / 2, self.HEIGHT / 2]) # Failsafe

    def _spawn_bug(self):
        pos = self._get_safe_spawn_pos()
        self.bugs.append({
            'pos': pos,
            'target': self._get_safe_spawn_pos(min_dist=0),
            'radius': 8,
            'anim_offset': self.np_random.uniform(0, 2 * math.pi)
        })

    def _spawn_trap(self):
        pos = self._get_safe_spawn_pos()
        self.traps.append({'pos': pos, 'radius': 15})

    # --- Effects ---
    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(10, 20),
                'color': color
            })

    def _create_squash_effect(self):
        self.squash_effects.append({
            'pos': self.cursor_pos.copy(),
            'life': 8,
            'max_life': 8,
            'radius': self.squash_radius
        })

    # --- Rendering ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_floor()
        self._render_traps()
        self._render_bugs()
        self._render_effects()
        self._render_cursor()

    def _generate_tile_map(self):
        tile_w, tile_h = 80, 40
        for r in range(-2, self.HEIGHT // tile_h + 2):
            for c in range(-2, self.WIDTH // tile_w + 2):
                cx = (c - r) * tile_w / 2
                cy = (c + r) * tile_h / 2
                color = self.COLOR_TILE_1 if (r + c) % 2 == 0 else self.COLOR_TILE_2
                points = [
                    (cx, cy - tile_h / 2), (cx + tile_w / 2, cy),
                    (cx, cy + tile_h / 2), (cx - tile_w / 2, cy)
                ]
                self.tile_map.append({'points': points, 'color': color})

    def _render_floor(self):
        for tile in self.tile_map:
            pygame.gfxdraw.filled_polygon(self.screen, [ (int(p[0]), int(p[1])) for p in tile['points']], tile['color'])

    def _render_traps(self):
        min_dist_trap = self._get_min_dist_to_entity(self.traps)
        for trap in self.traps:
            x, y = int(trap['pos'][0]), int(trap['pos'][1])
            r = trap['radius']
            
            # Glow effect when cursor is near
            dist_to_cursor = np.linalg.norm(self.cursor_pos - trap['pos'])
            glow_alpha = max(0, 1 - dist_to_cursor / (self.trap_proximity_threshold * 2))
            if glow_alpha > 0.1:
                glow_r = int(r * (1.5 + 0.5 * math.sin(self.steps * 0.2)))
                glow_color = (*self.COLOR_TRAP_GLOW, int(100 * glow_alpha))
                surf = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
                pygame.draw.circle(surf, glow_color, (glow_r, glow_r), glow_r)
                self.screen.blit(surf, (x - glow_r, y - glow_r), special_flags=pygame.BLEND_RGBA_ADD)

            pygame.gfxdraw.filled_circle(self.screen, x, y, r, self.COLOR_TRAP_BASE)
            pygame.gfxdraw.aacircle(self.screen, x, y, r, self.COLOR_TRAP_GLOW)

    def _render_bugs(self):
        for bug in self.bugs:
            x, y = int(bug['pos'][0]), int(bug['pos'][1])
            r = bug['radius']
            # Body
            pygame.gfxdraw.filled_circle(self.screen, x, y, r, self.COLOR_BUG_OUTER)
            pygame.gfxdraw.filled_circle(self.screen, x, y, r-3, self.COLOR_BUG_INNER)
            # Legs
            for i in range(6):
                angle = (i * math.pi / 3) + math.sin(bug['anim_offset'] + i) * 0.3
                start_pos = (
                    x + (r - 2) * math.cos(angle),
                    y + (r - 2) * math.sin(angle)
                )
                end_pos = (
                    x + (r + 4) * math.cos(angle),
                    y + (r + 4) * math.sin(angle)
                )
                pygame.draw.line(self.screen, self.COLOR_BUG_INNER, start_pos, end_pos, 1)

    def _render_effects(self):
        for p in self.particles:
            size = int(p['life'] / 4)
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], p['pos'].astype(int), size)
        
        for e in self.squash_effects:
            progress = (e['max_life'] - e['life']) / e['max_life']
            current_radius = int(e['radius'] * progress)
            alpha = int(255 * (1 - progress))
            if alpha > 0 and current_radius > 0:
                color = (*self.COLOR_BUG_INNER, alpha)
                surf = pygame.Surface((current_radius * 2, current_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (current_radius, current_radius), current_radius)
                self.screen.blit(surf, (int(e['pos'][0]) - current_radius, int(e['pos'][1]) - current_radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_cursor(self):
        x, y = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        min_dist_trap = self._get_min_dist_to_entity(self.traps)
        
        color = self.COLOR_CURSOR
        if min_dist_trap < self.trap_proximity_threshold:
            p = min_dist_trap / self.trap_proximity_threshold
            color = (
                int(self.COLOR_CURSOR[0] * p + self.COLOR_CURSOR_DANGER[0] * (1 - p)),
                int(self.COLOR_CURSOR[1] * p + self.COLOR_CURSOR_DANGER[1] * (1 - p)),
                int(self.COLOR_CURSOR[2] * p + self.COLOR_CURSOR_DANGER[2] * (1 - p)),
            )
        
        pulse = 0.5 * (1 + math.sin(self.steps * 0.25))
        pygame.gfxdraw.aacircle(self.screen, x, y, 10, color)
        pygame.gfxdraw.filled_circle(self.screen, x, y, 3 + int(pulse * 3), color)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SQUASHED: {self.bugs_squashed} / {self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        # Traps
        trap_text = self.font_small.render(f"TRAPS: {self.traps_triggered} / {self.LOSE_CONDITION}", True, self.COLOR_TEXT)
        self.screen.blit(trap_text, (self.WIDTH - trap_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            msg = "VICTORY!" if self.win else "GAME OVER"
            color = self.COLOR_BUG_INNER if self.win else self.COLOR_TRAP_GLOW
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    # --- Gymnasium Interface ---
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "bugs_squashed": self.bugs_squashed,
            "traps_triggered": self.traps_triggered,
        }
        
    def _get_min_dist_to_entity(self, entities):
        if not entities:
            return float('inf')
        dists = [np.linalg.norm(self.cursor_pos - e['pos']) for e in entities]
        return min(dists)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Bug Squash")
    
    running = True
    total_reward = 0
    
    # Map Pygame keys to MultiDiscrete actions
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    while running:
        movement = 0
        space_pressed = 0
        shift_pressed = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
        
        keys = pygame.key.get_pressed()
        
        for key, move_action in key_map.items():
            if keys[key]:
                movement = move_action
                break # Prioritize one movement key
                
        if keys[pygame.K_SPACE]:
            space_pressed = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_pressed = 1
            
        action = [movement, space_pressed, shift_pressed]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}")
            print("Press 'R' to restart.")
            
        env.clock.tick(30) # Run at 30 FPS
        
    env.close()