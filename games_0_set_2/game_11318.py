import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:03:25.514658
# Source Brief: brief_01318.md
# Brief Index: 1318
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A rhythm-based tower defense game. Place and upgrade towers to deflect incoming 'notes' and protect your stage."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to place a tower and shift to upgrade an existing tower."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        # Colors (Vibrant Neon on Dark)
        self.COLOR_BG = (15, 10, 25)
        self.COLOR_GRID = (30, 20, 50)
        self.COLOR_CURSOR = (255, 255, 255, 100)
        self.COLOR_TOWER_L1 = (0, 255, 255)
        self.COLOR_TOWER_L2 = (255, 0, 255)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_HEALTH_BAR_BG = (80, 20, 40)
        self.COLOR_HEALTH_BAR_FG = (255, 40, 100)
        self.NOTE_COLORS = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100)
        ]

        # Game Parameters
        self.MAX_HEALTH = 100
        self.INITIAL_RESOURCES = 50
        self.MAX_STEPS = 1000
        self.CURSOR_SPEED = 10
        self.TOWER_COST = 25
        self.UPGRADE_COST = 40
        self.TOWER_RADIUS_L1 = 12
        self.TOWER_RADIUS_L2 = 16
        self.NOTE_RADIUS = 6
        self.NOTE_BASE_SPEED = 2.0
        self.NOTE_INITIAL_SPAWN_RATE = 60 # Steps per spawn

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_big = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # --- State Variables ---
        # These are initialized in reset()
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.stage_health = None
        self.resources = None
        self.cursor_pos = None
        self.notes = None
        self.towers = None
        self.particles = None
        self.note_spawn_timer = None
        self.note_current_spawn_rate = None
        self.note_current_speed = None
        self.prev_space_held = None
        self.prev_shift_held = None

        # self.reset() is called by the wrapper or user
        # self.validate_implementation() is for debugging, not needed in final env

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # Seed python's random module
            random.seed(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.stage_health = self.MAX_HEALTH
        self.resources = self.INITIAL_RESOURCES
        self.cursor_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        
        self.notes = []
        self.towers = []
        self.particles = []

        self.note_spawn_timer = 0
        self.note_current_spawn_rate = self.NOTE_INITIAL_SPAWN_RATE
        self.note_current_speed = self.NOTE_BASE_SPEED

        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # --- 1. Handle Input & Action-based Rewards ---
        reward += self._handle_input(action)

        # --- 2. Update Game Logic & Event-based Rewards ---
        self._update_difficulty()
        self._spawn_notes()
        
        update_reward = self._update_game_entities()
        reward += update_reward

        self.steps += 1
        
        # --- 3. Check Termination & Terminal Rewards ---
        terminated = self.stage_health <= 0 or self.steps >= self.MAX_STEPS
        truncated = False # No specific truncation condition besides max steps
        if self.steps >= self.MAX_STEPS and self.stage_health > 0:
            self.win = True
            reward += 100.0
            self.score += 1000 # Bonus score for winning
            terminated = True
        elif self.stage_health <= 0:
            reward -= 100.0
            terminated = True
        
        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        action_reward = 0.0

        # Move cursor
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED # Up
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED # Down
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED # Left
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT - 40) # Keep out of damage zone

        # Place tower on space PRESS
        if space_held and not self.prev_space_held:
            action_reward += self._place_tower()

        # Upgrade tower on shift PRESS
        if shift_held and not self.prev_shift_held:
            action_reward += self._upgrade_tower()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        return action_reward

    def _place_tower(self):
        if self.resources >= self.TOWER_COST:
            # Check for existing tower proximity
            for tower in self.towers:
                dist = np.linalg.norm(self.cursor_pos - tower['pos'])
                if dist < self.TOWER_RADIUS_L1 * 2:
                    # SFX: action_fail
                    return 0.0 # Cannot place too close

            self.resources -= self.TOWER_COST
            self.towers.append({
                'pos': self.cursor_pos.copy(),
                'level': 1,
                'radius': self.TOWER_RADIUS_L1,
                'pulse': 0.0
            })
            self._create_particles(self.cursor_pos, 20, self.COLOR_TOWER_L1)
            # SFX: place_tower
            return -0.01 * self.TOWER_COST # Reward for spending resources
        # SFX: action_fail
        return 0.0

    def _upgrade_tower(self):
        for tower in self.towers:
            dist = np.linalg.norm(self.cursor_pos - tower['pos'])
            if dist < tower['radius'] and tower['level'] == 1:
                if self.resources >= self.UPGRADE_COST:
                    self.resources -= self.UPGRADE_COST
                    tower['level'] = 2
                    tower['radius'] = self.TOWER_RADIUS_L2
                    self._create_particles(tower['pos'], 40, self.COLOR_TOWER_L2, 2.0)
                    # SFX: upgrade_tower
                    self.score += 50
                    return 5.0 - (0.01 * self.UPGRADE_COST) # Event reward + spending cost
        # SFX: action_fail
        return 0.0

    def _update_difficulty(self):
        # Note speed increases by 0.5 every 100 steps
        self.note_current_speed = self.NOTE_BASE_SPEED + 0.5 * (self.steps // 100)
        # Spawn rate also increases
        self.note_current_spawn_rate = max(20, self.NOTE_INITIAL_SPAWN_RATE - (self.steps // 50) * 2)

    def _spawn_notes(self):
        self.note_spawn_timer -= 1
        if self.note_spawn_timer <= 0:
            self.note_spawn_timer = self.note_current_spawn_rate
            
            spawn_x = random.uniform(self.NOTE_RADIUS, self.WIDTH - self.NOTE_RADIUS)
            pos = np.array([spawn_x, -self.NOTE_RADIUS])
            angle = random.uniform(math.pi * 0.25, math.pi * 0.75) # Downward angle
            vel = np.array([math.cos(angle), math.sin(angle)]) * self.note_current_speed
            
            self.notes.append({
                'pos': pos,
                'vel': vel,
                'color': random.choice(self.NOTE_COLORS),
                'trail': deque(maxlen=10)
            })

    def _update_game_entities(self):
        reward = 0.0
        
        # Update towers (for animation)
        for tower in self.towers:
            tower['pulse'] = (tower['pulse'] + 0.05 * tower['level']) % (2 * math.pi)

        # Update notes
        notes_to_remove = []
        for i, note in enumerate(self.notes):
            note['trail'].append(note['pos'].copy())
            note['pos'] += note['vel']

            # Wall collisions
            if note['pos'][0] <= self.NOTE_RADIUS or note['pos'][0] >= self.WIDTH - self.NOTE_RADIUS:
                note['vel'][0] *= -1
                note['pos'][0] = np.clip(note['pos'][0], self.NOTE_RADIUS, self.WIDTH - self.NOTE_RADIUS)
                # SFX: bounce_wall
            if note['pos'][1] <= self.NOTE_RADIUS:
                note['vel'][1] *= -1
                note['pos'][1] = np.clip(note['pos'][1], self.NOTE_RADIUS, self.HEIGHT)
                # SFX: bounce_wall
            
            # Tower collisions
            for tower in self.towers:
                dist_vec = note['pos'] - tower['pos']
                dist = np.linalg.norm(dist_vec)
                if dist < self.NOTE_RADIUS + tower['radius']:
                    # SFX: bounce_tower
                    self._create_particles(note['pos'], 10, note['color'])
                    self.resources += 1
                    self.score += 10
                    reward += 0.1 # Deflection reward

                    # Bounce physics
                    normal = dist_vec / dist
                    note['vel'] = note['vel'] - 2 * np.dot(note['vel'], normal) * normal
                    note['vel'] *= 1.05 # Speed up on deflection
                    note['pos'] = tower['pos'] + normal * (self.NOTE_RADIUS + tower['radius']) # Prevent sticking

            # Note-Note collisions
            for j in range(i + 1, len(self.notes)):
                other_note = self.notes[j]
                dist_vec = note['pos'] - other_note['pos']
                dist = np.linalg.norm(dist_vec)
                if dist < self.NOTE_RADIUS * 2:
                    # Simplified elastic collision
                    normal = dist_vec / dist
                    v1_p = np.dot(note['vel'], normal)
                    v2_p = np.dot(other_note['vel'], normal)
                    note['vel'] += normal * (v2_p - v1_p)
                    other_note['vel'] += normal * (v1_p - v2_p)
                    self._create_particles(note['pos'], 5, (200, 200, 200))
                    # Note: No direct reward for note-note collision to keep it simple

            # Stage damage
            if note['pos'][1] >= self.HEIGHT - self.NOTE_RADIUS:
                self.stage_health -= 10
                self.stage_health = max(0, self.stage_health)
                notes_to_remove.append(i)
                self._create_particles(note['pos'], 30, self.COLOR_HEALTH_BAR_FG, 2.5)
                # SFX: stage_hit
        
        # Remove notes that hit the stage
        for i in sorted(notes_to_remove, reverse=True):
            del self.notes[i]

        # Update particles
        particles_to_remove = []
        for i, p in enumerate(self.particles):
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                particles_to_remove.append(i)
        for i in sorted(particles_to_remove, reverse=True):
            del self.particles[i]
            
        return reward

    def _create_particles(self, pos, count, color, speed_mult=1.0):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2.0) * speed_mult
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': random.randint(15, 30),
                'color': color
            })
            
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_particles()
        self._render_towers()
        self._render_notes()
        self._render_cursor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)
        # Damage zone
        s = pygame.Surface((self.WIDTH, 40))
        s.set_alpha(50)
        s.fill(self.COLOR_HEALTH_BAR_FG)
        self.screen.blit(s, (0, self.HEIGHT - 40))

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30.0))
            color_with_alpha = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 2, color_with_alpha)
            
    def _render_towers(self):
        for t in self.towers:
            pos_int = (int(t['pos'][0]), int(t['pos'][1]))
            radius = int(t['radius'])
            color = self.COLOR_TOWER_L1 if t['level'] == 1 else self.COLOR_TOWER_L2
            
            # Pulsing aura
            pulse_radius = radius + 4 + 3 * math.sin(t['pulse'])
            pulse_alpha = 50 + 40 * math.sin(t['pulse'])
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(pulse_radius), (*color, int(pulse_alpha)))
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], int(pulse_radius), (*color, int(pulse_alpha)))
            
            # Main tower shape
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius, color)

    def _render_notes(self):
        for n in self.notes:
            # Trail
            if len(n['trail']) > 1:
                for i, pos in enumerate(n['trail']):
                    alpha = int(255 * (i / len(n['trail'])) * 0.5)
                    pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), self.NOTE_RADIUS, (*n['color'], alpha))
            # Note head
            pos_int = (int(n['pos'][0]), int(n['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.NOTE_RADIUS, n['color'])
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.NOTE_RADIUS, n['color'])

    def _render_cursor(self):
        pos_int = (int(self.cursor_pos[0]), int(self.cursor_pos[1]))
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.TOWER_RADIUS_L1, self.COLOR_CURSOR)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (pos_int[0] - 5, pos_int[1]), (pos_int[0] + 5, pos_int[1]))
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (pos_int[0], pos_int[1] - 5), (pos_int[0], pos_int[1] + 5))

    def _render_ui(self):
        # Health Bar
        health_ratio = self.stage_health / self.MAX_HEALTH
        bar_width = self.WIDTH - 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (10, 10, int(bar_width * health_ratio), 20))

        # Resources
        res_text = self.font_main.render(f"RES: {self.resources}", True, self.COLOR_TEXT)
        self.screen.blit(res_text, (15, 35))

        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 15, 35))

        # Game Over / Win Text
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "VICTORY" if self.win else "GAME OVER"
            color = (100, 255, 150) if self.win else (255, 100, 100)
            end_text = self.font_big.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage_health": self.stage_health,
            "resources": self.resources,
            "towers": len(self.towers),
            "notes": len(self.notes),
        }
        
    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To use, you might need to remove the SDL_VIDEODRIVER dummy setting
    # and install pygame: pip install pygame
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Rhythm Tower Defense")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    while not done:
        # --- Manual Control Mapping ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # --- Pygame Rendering ---
        # The observation is already a rendered surface, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & FPS ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # Run at 30 FPS
        pygame.display.set_caption(f"Score: {info['score']} | Health: {info['stage_health']}")

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()