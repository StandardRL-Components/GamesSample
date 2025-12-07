import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:51:06.527463
# Source Brief: brief_03043.md
# Brief Index: 3043
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper classes for game entities
class Whale:
    def __init__(self, pos, anim_offset):
        self.pos = np.array(pos, dtype=np.float32)
        self.health = 100
        self.max_health = 100
        self.size = 15
        self.anim_offset = anim_offset
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)

    def update(self, currents, screen_dims):
        # Apply current forces
        force = np.array([0.0, 0.0], dtype=np.float32)
        for current in currents:
            dist_sq = np.sum((self.pos - current['pos'])**2)
            if dist_sq < current['radius']**2:
                force += current['dir'] * current['strength']
        
        # Apply force and damping
        self.velocity += force
        self.velocity *= 0.9  # Damping

        # Move whale
        self.pos += self.velocity
        
        # Keep within bounds
        self.pos[0] = np.clip(self.pos[0], self.size, screen_dims[0] - self.size)
        self.pos[1] = np.clip(self.pos[1], self.size, screen_dims[1] - self.size)

    def draw(self, surface, steps):
        # Body bobbing animation
        bob = math.sin(steps * 0.1 + self.anim_offset) * 3
        body_pos = (int(self.pos[0]), int(self.pos[1] + bob))
        
        # Tail flapping animation
        angle = math.sin(steps * 0.2 + self.anim_offset) * 0.5
        tail_length = self.size * 0.8
        tail_end_pos = (
            body_pos[0] - tail_length * math.cos(angle),
            body_pos[1] + tail_length * math.sin(angle)
        )
        
        # Draw body (filled circle with outline)
        pygame.gfxdraw.filled_circle(surface, body_pos[0], body_pos[1], int(self.size), (230, 240, 255))
        pygame.gfxdraw.aacircle(surface, body_pos[0], body_pos[1], int(self.size), (255, 255, 255))
        
        # Draw tail
        pygame.draw.line(surface, (255, 255, 255), body_pos, tail_end_pos, 3)
        
        # Draw eye
        pygame.gfxdraw.filled_circle(surface, body_pos[0] + int(self.size/2), body_pos[1], 2, (0, 0, 0))

class Threat:
    def __init__(self, pos, speed):
        self.pos = np.array(pos, dtype=np.float32)
        self.size = 10
        self.speed = speed
        self.anim_offset = random.uniform(0, 2 * math.pi)

    def update(self, pod_center, currents):
        # Apply current forces (threats are less affected)
        force = np.array([0.0, 0.0], dtype=np.float32)
        for current in currents:
            dist_sq = np.sum((self.pos - current['pos'])**2)
            if dist_sq < current['radius']**2:
                force += current['dir'] * current['strength'] * 0.3 # Less effect
        
        direction_to_pod = pod_center - self.pos
        dist = np.linalg.norm(direction_to_pod)
        if dist > 0:
            direction_to_pod /= dist
        
        # Sine wave movement towards pod
        t = self.anim_offset + pygame.time.get_ticks() * 0.002
        perpendicular = np.array([-direction_to_pod[1], direction_to_pod[0]])
        wave_offset = perpendicular * math.sin(t) * 0.5
        
        final_direction = direction_to_pod + wave_offset
        norm = np.linalg.norm(final_direction)
        if norm > 0:
            final_direction /= norm

        self.pos += final_direction * self.speed + force

    def draw(self, surface):
        # Draw shark fin
        p1 = (self.pos[0], self.pos[1] - self.size)
        p2 = (self.pos[0] - self.size, self.pos[1] + self.size)
        p3 = (self.pos[0] + self.size, self.pos[1] + self.size)
        points = [p1, p2, p3]
        
        pygame.gfxdraw.filled_trigon(surface, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), (200, 50, 50))
        pygame.gfxdraw.aatrigon(surface, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), (255, 80, 80))

class Particle:
    def __init__(self, pos, vel, life, color, size):
        self.pos = np.array(pos, dtype=np.float32)
        self.vel = np.array(vel, dtype=np.float32)
        self.life = life
        self.max_life = life
        self.color = color
        self.size = size

    def update(self):
        self.pos += self.vel
        self.life -= 1

    def draw(self, surface):
        alpha = int(255 * (self.life / self.max_life))
        color = (*self.color, alpha)
        pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), int(self.size), color)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Guide a pod of migrating whales to safety. Spell words to create currents, barriers, "
        "and other oceanic effects to protect them from threats."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. "
        "Press space to select letters and spell words. Press shift to use unlocked abilities."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    UI_HEIGHT = 100
    GAME_HEIGHT = SCREEN_HEIGHT - UI_HEIGHT
    
    COLOR_BG = (10, 20, 40)
    COLOR_TEXT = (220, 220, 255)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_TILE_BG = (30, 50, 80)
    COLOR_TILE_BORDER = (50, 80, 120)
    COLOR_WORD_BOX = (20, 40, 70)
    COLOR_SUCCESS = (100, 255, 100)
    COLOR_FAIL = (255, 100, 100)

    VOWELS = "AEIOU"
    CONSONANTS = "BCDFGHJKLMNPQRSTVWXYZ"
    
    WORD_LIST = {
        "SEA": {"type": "current", "dir": (1, 0), "strength": 0.3},
        "TIDE": {"type": "current", "dir": (-1, 0), "strength": 0.3},
        "RISE": {"type": "current", "dir": (0, -1), "strength": 0.3},
        "SINK": {"type": "current", "dir": (0, 1), "strength": 0.3},
        "WALL": {"type": "barrier", "duration": 150},
        "GALE": {"type": "summon", "name": "whirlpool"},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont('Consolas', 18)
        self.font_medium = pygame.font.SysFont('Consolas', 24)
        self.font_large = pygame.font.SysFont('Consolas', 32)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.whales = []
        self.threats = []
        self.currents = []
        self.particles = []
        self.barriers = []
        
        self.tile_grid = []
        self.cursor_pos = [0, 0]
        self.current_word = ""
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.threat_spawn_timer = 0
        self.initial_threat_speed = 0.5
        self.current_threat_speed = self.initial_threat_speed
        
        self.word_feedback_timer = 0
        self.word_feedback_text = ""
        self.word_feedback_color = self.COLOR_TEXT
        
        self.unlocked_summons = {"whirlpool": False}
        self.summon_cooldown = 0

        self.destination_y = 40
        
        # self.reset() # reset is called by the wrapper
        # self.validate_implementation() # validation is done by the test suite
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Whales
        self.whales = [Whale([self.SCREEN_WIDTH / 2 + (i-1.5)*40, self.GAME_HEIGHT - 50], random.uniform(0, 2*math.pi)) for i in range(3)]
        
        # Threats & Environment
        self.threats = []
        self.currents = []
        self.particles = []
        self.barriers = []
        self.threat_spawn_timer = 100
        self.current_threat_speed = self.initial_threat_speed
        
        # Word & Tile state
        self.tile_grid_dims = (2, 8)
        self._refill_tiles()
        self.cursor_pos = [0, 0]
        self.current_word = ""
        
        # Action state
        self.prev_space_held = True # Prevent action on first frame
        self.prev_shift_held = True
        
        # UI state
        self.word_feedback_timer = 0
        self.unlocked_summons = {"whirlpool": False}
        self.summon_cooldown = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1
        
        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held
        
        # 1. Cursor movement
        if movement != 0:
            # In MultiDiscrete, 0 is no-op. The actions are 1-4.
            if movement == 1: self.cursor_pos[1] = (self.cursor_pos[1] - 1) % self.tile_grid_dims[1] # Left
            elif movement == 2: self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.tile_grid_dims[1] # Right
            elif movement == 3: self.cursor_pos[0] = (self.cursor_pos[0] - 1) % self.tile_grid_dims[0] # Up
            elif movement == 4: self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.tile_grid_dims[0] # Down
        
        # 2. Place tile
        if space_press:
            r, c = self.cursor_pos
            if self.tile_grid[r][c] is not None:
                self.current_word += self.tile_grid[r][c]
                self.tile_grid[r][c] = None
                
                # Check if word is complete
                if self.current_word in self.WORD_LIST:
                    reward += 0.1
                    self._trigger_word_effect(self.current_word)
                    self.current_word = ""
                elif len(self.current_word) >= 8: # Max word length
                    reward -= 0.1
                    self._show_word_feedback("Invalid Word", self.COLOR_FAIL)
                    self.current_word = ""

        # 3. Use summon
        if shift_press and self.summon_cooldown <= 0 and self.unlocked_summons["whirlpool"]:
            self._create_whirlpool()
            self.summon_cooldown = 300 # 10 seconds at 30fps
        
        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        # --- Game Logic Update ---
        self._update_progression()
        self._update_spawns()
        self._update_currents()
        self._update_barriers()
        
        pod_center = self._get_pod_center()
        for whale in self.whales:
            whale.update(self.currents, (self.SCREEN_WIDTH, self.GAME_HEIGHT))
        
        for threat in self.threats:
            old_pos = threat.pos.copy()
            threat.update(pod_center, self.currents)
            
            # Check for deflection reward
            old_dist_to_pod = np.linalg.norm(old_pos - pod_center)
            new_dist_to_pod = np.linalg.norm(threat.pos - pod_center)
            if new_dist_to_pod > old_dist_to_pod + 0.1: # If pushed away
                reward += 1.0

        reward += self._handle_collisions()
        self._update_particles()
        self._refill_tiles()
        if self.summon_cooldown > 0: self.summon_cooldown -= 1
        if self.word_feedback_timer > 0: self.word_feedback_timer -= 1
        
        # --- Termination Check ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if all(w.health <= 0 for w in self.whales):
                reward -= 100
            elif self._get_pod_center()[1] <= self.destination_y:
                reward += 100
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    # --- Update Helpers ---
    def _update_progression(self):
        if self.steps > 0 and self.steps % 200 == 0:
            self.current_threat_speed += 0.05
        if self.score >= 50 and not self.unlocked_summons["whirlpool"]:
            self.unlocked_summons["whirlpool"] = True
            self._show_word_feedback("GALE unlocked!", self.COLOR_SUCCESS, 120)

    def _update_spawns(self):
        self.threat_spawn_timer -= 1
        if self.threat_spawn_timer <= 0:
            spawn_x = random.choice([0, self.SCREEN_WIDTH])
            spawn_y = random.uniform(0, self.GAME_HEIGHT * 0.75)
            self.threats.append(Threat([spawn_x, spawn_y], self.current_threat_speed))
            self.threat_spawn_timer = max(30, 150 - self.steps // 20)

    def _update_currents(self):
        self.currents = [c for c in self.currents if c['life'] > 0]
        for c in self.currents:
            c['life'] -= 1
            if c['life'] % 3 == 0:
                p_pos = c['pos'] + np.random.normal(0, c['radius']/2, 2)
                p_vel = c['dir'] * 2
                self.particles.append(Particle(p_pos, p_vel, 20, (100, 200, 255), random.uniform(1,3)))

    def _update_barriers(self):
        self.barriers = [b for b in self.barriers if b['life'] > 0]
        for b in self.barriers:
            b['life'] -= 1

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

    def _handle_collisions(self):
        reward = 0
        
        # Threats vs Whales
        for threat in self.threats[:]:
            for whale in self.whales:
                if whale.health > 0:
                    dist = np.linalg.norm(threat.pos - whale.pos)
                    if dist < threat.size + whale.size:
                        whale.health -= 25
                        reward -= 5
                        if threat in self.threats: self.threats.remove(threat)
                        for _ in range(20):
                            angle = random.uniform(0, 2*math.pi)
                            vel = np.array([math.cos(angle), math.sin(angle)]) * random.uniform(1, 4)
                            self.particles.append(Particle(whale.pos, vel, 30, (255, 80, 80), random.uniform(1,4)))
                        break
        
        # Threats vs Barriers
        for threat in self.threats[:]:
            for barrier in self.barriers:
                if barrier['rect'].collidepoint(threat.pos):
                    if threat in self.threats: self.threats.remove(threat)
                    for _ in range(15):
                        angle = random.uniform(0, 2*math.pi)
                        vel = np.array([math.cos(angle), math.sin(angle)]) * random.uniform(1, 3)
                        self.particles.append(Particle(threat.pos, vel, 20, (200, 200, 100), random.uniform(1,3)))
                    break
                    
        return reward

    def _check_termination(self):
        if self.steps >= 5000:
            return True
        if all(w.health <= 0 for w in self.whales):
            return True
        if self._get_pod_center()[1] <= self.destination_y:
            return True
        return False

    def _refill_tiles(self):
        if not self.tile_grid:
            self.tile_grid = [[None for _ in range(self.tile_grid_dims[1])] for _ in range(self.tile_grid_dims[0])]
        
        for r in range(self.tile_grid_dims[0]):
            for c in range(self.tile_grid_dims[1]):
                if self.tile_grid[r][c] is None:
                    # Ensure at least a few vowels
                    if random.random() < 0.4 or sum(row.count(v) for row in self.tile_grid if row for v in self.VOWELS) < 3:
                        self.tile_grid[r][c] = random.choice(self.VOWELS)
                    else:
                        self.tile_grid[r][c] = random.choice(self.CONSONANTS)

    # --- Word & Effect Helpers ---
    def _trigger_word_effect(self, word):
        effect = self.WORD_LIST[word]
        self._show_word_feedback(f"{word}!", self.COLOR_SUCCESS)
        self.score += len(word) * 10
        
        if effect["type"] == "current":
            pod_center = self._get_pod_center()
            self.currents.append({
                'pos': pod_center,
                'dir': np.array(effect['dir'], dtype=np.float32),
                'strength': effect['strength'],
                'radius': 150,
                'life': 100
            })
        elif effect["type"] == "barrier":
            pod_center = self._get_pod_center()
            rect = pygame.Rect(pod_center[0] - 75, pod_center[1] - 100, 150, 20)
            self.barriers.append({'rect': rect, 'life': effect['duration']})
        elif effect["type"] == "summon" and effect["name"] == "whirlpool":
            if self.unlocked_summons["whirlpool"]:
                self._create_whirlpool()
            else:
                self._show_word_feedback("GALE not yet unlocked!", self.COLOR_FAIL)

    def _create_whirlpool(self):
        pod_center = self._get_pod_center()
        # A whirlpool is just a strong, circular current
        self.currents.append({
            'pos': pod_center + np.array([0, -100]),
            'dir': np.array([0,0]), # Special case handled in threat update logic
            'strength': -0.5, # Negative strength pulls in
            'radius': 180,
            'life': 200,
            'is_whirlpool': True
        })

    def _show_word_feedback(self, text, color, duration=60):
        self.word_feedback_text = text
        self.word_feedback_color = color
        self.word_feedback_timer = duration

    # --- Rendering ---
    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_details()
        self._render_particles()
        self._render_currents()
        self._render_barriers()
        self._render_threats()
        self._render_whales()
        self._render_ui()

    def _render_background_details(self):
        # Migration route
        for y in range(int(self.destination_y), self.GAME_HEIGHT, 20):
            pygame.gfxdraw.line(self.screen, self.SCREEN_WIDTH//2 - 5, y, self.SCREEN_WIDTH//2 + 5, y, (20, 40, 60))
        pygame.gfxdraw.filled_circle(self.screen, self.SCREEN_WIDTH//2, self.destination_y, 10, (100, 255, 100))
        pygame.gfxdraw.aacircle(self.screen, self.SCREEN_WIDTH//2, self.destination_y, 10, (200, 255, 200))
        
        # Random background bubbles
        for i in range(10):
            x = (hash(i*10 + self.steps//10) % self.SCREEN_WIDTH)
            y = (hash(i*20 + self.steps//10) % self.GAME_HEIGHT)
            r = (hash(i*30 + self.steps//10) % 3) + 1
            pygame.gfxdraw.aacircle(self.screen, x, y, r, (20, 40, 60, 100))

    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen)

    def _render_currents(self):
        for c in self.currents:
            if c.get('is_whirlpool'):
                life_ratio = c['life'] / 200.0
                for i in range(5):
                    radius = c['radius'] * life_ratio * (i/4.0)
                    angle = (self.steps * 0.1 + i) % (2*math.pi)
                    alpha = int(100 * life_ratio)
                    pygame.gfxdraw.arc(self.screen, int(c['pos'][0]), int(c['pos'][1]), int(radius), int(math.degrees(angle)), int(math.degrees(angle))+180, (150, 220, 255, alpha))


    def _render_barriers(self):
        for b in self.barriers:
            alpha = int(255 * (b['life'] / 150.0))
            color = (*self.COLOR_SUCCESS[:3], alpha)
            pygame.draw.rect(self.screen, color, b['rect'], border_radius=5)

    def _render_whales(self):
        for whale in self.whales:
            if whale.health > 0:
                whale.draw(self.screen, self.steps)
                # Health bar
                health_ratio = whale.health / whale.max_health
                bar_color = (int(255 * (1 - health_ratio)), int(255 * health_ratio), 0)
                bar_pos = (whale.pos[0] - whale.size, whale.pos[1] - whale.size - 8)
                bar_dims = (whale.size * 2, 5)
                pygame.draw.rect(self.screen, (50,50,50), (*bar_pos, *bar_dims))
                pygame.draw.rect(self.screen, bar_color, (*bar_pos, bar_dims[0] * health_ratio, bar_dims[1]))

    def _render_threats(self):
        for threat in self.threats:
            threat.draw(self.screen)

    def _render_ui(self):
        # UI Background
        ui_rect = pygame.Rect(0, self.GAME_HEIGHT, self.SCREEN_WIDTH, self.UI_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_WORD_BOX, ui_rect)
        pygame.draw.line(self.screen, self.COLOR_TILE_BORDER, (0, self.GAME_HEIGHT), (self.SCREEN_WIDTH, self.GAME_HEIGHT), 2)
        
        # Tiles
        tile_w, tile_h = 40, 40
        start_x = (self.SCREEN_WIDTH - self.tile_grid_dims[1] * tile_w) / 2
        start_y = self.GAME_HEIGHT + 10
        for r in range(self.tile_grid_dims[0]):
            for c in range(self.tile_grid_dims[1]):
                rect = pygame.Rect(start_x + c*tile_w, start_y + r*tile_h, tile_w-2, tile_h-2)
                pygame.draw.rect(self.screen, self.COLOR_TILE_BG, rect, border_radius=3)
                pygame.draw.rect(self.screen, self.COLOR_TILE_BORDER, rect, 1, border_radius=3)
                
                letter = self.tile_grid[r][c]
                if letter:
                    text_surf = self.font_medium.render(letter, True, self.COLOR_TEXT)
                    text_rect = text_surf.get_rect(center=rect.center)
                    self.screen.blit(text_surf, text_rect)
        
        # Cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = pygame.Rect(start_x + cursor_c * tile_w, start_y + cursor_r * tile_h, tile_w-2, tile_h-2)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2, border_radius=3)
        
        # Current word
        word_text_surf = self.font_large.render(self.current_word, True, self.COLOR_TEXT)
        word_text_rect = word_text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.GAME_HEIGHT - 25))
        self.screen.blit(word_text_surf, word_text_rect)

        # Word feedback
        if self.word_feedback_timer > 0:
            alpha = int(255 * (self.word_feedback_timer / 60.0))
            feedback_surf = self.font_medium.render(self.word_feedback_text, True, self.word_feedback_color)
            feedback_surf.set_alpha(alpha)
            feedback_rect = feedback_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.GAME_HEIGHT / 2))
            self.screen.blit(feedback_surf, feedback_rect)
        
        # Score and Steps
        score_surf = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))
        steps_surf = self.font_small.render(f"Steps: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(steps_surf, (10, 35))

        # Summon status
        gale_color = self.COLOR_SUCCESS if self.unlocked_summons["whirlpool"] else (80,80,80)
        gale_text = "GALE [SHIFT]"
        if self.summon_cooldown > 0:
            gale_text = f"GALE ({self.summon_cooldown//30}s)"
            gale_color = (150,150,150)
        
        summon_surf = self.font_medium.render(gale_text, True, gale_color)
        summon_rect = summon_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(summon_surf, summon_rect)

    # --- Getters ---
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "pod_health": sum(w.health for w in self.whales),
            "threat_count": len(self.threats)
        }

    def _get_pod_center(self):
        if not self.whales or all(w.health <= 0 for w in self.whales):
            return np.array([self.SCREEN_WIDTH/2, self.GAME_HEIGHT/2])
        
        alive_whales = [w.pos for w in self.whales if w.health > 0]
        return np.mean(alive_whales, axis=0) if alive_whales else np.array([0,0])

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Key mapping for manual play
    # Action: 0:No-op, 1:Left, 2:Right, 3:Up, 4:Down
    key_map = {
        pygame.K_LEFT: 1,
        pygame.K_RIGHT: 2,
        pygame.K_UP: 3,
        pygame.K_DOWN: 4,
    }

    # Use a display for manual play
    pygame.display.set_caption("Whale Migration")
    real_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    total_reward = 0
    while not done:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        for key, move_action in key_map.items():
            if keys[key]:
                movement = move_action
                break # Process one movement key at a time
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
        if keys[pygame.K_r]:
            obs, info = env.reset()
            total_reward = 0
            print("--- Game Reset ---")

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0,0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()