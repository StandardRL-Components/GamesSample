import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:56:07.406601
# Source Brief: brief_00714.md
# Brief Index: 714
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "As a nanobot, fight a viral infection by matching your attack types to virus weaknesses. "
        "Destroy all viruses before they overwhelm you."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press SHIFT to cycle attacks and SPACE to fire at the nearest virus."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 1500
    PLAYER_SPEED = 4.0
    PLAYER_RADIUS = 12
    PLAYER_MAX_HEALTH = 100
    VIRUS_RADIUS = 15
    VIRUS_MAX_HEALTH = 3
    MAX_VIRUSES = 10
    ATTACK_RANGE = 150
    ATTACK_COOLDOWN = 15  # steps
    PARTICLE_LIFESPAN = 20
    
    # Word list for attacks/weaknesses
    WORD_LIST = ["LYSATE", "PHAGE", "INHIBIT", "OXIDIZE", "BIND"]
    
    # Colors
    COLOR_BG = (18, 5, 20)
    COLOR_BG_CELLS = (40, 10, 30, 50)
    COLOR_PLAYER = (0, 180, 255)
    COLOR_PLAYER_GLOW = (0, 100, 200, 70)
    COLOR_VIRUS_WEAKNESSES = {
        "LYSATE": (255, 80, 80),   # Red
        "PHAGE": (80, 255, 80),    # Green
        "INHIBIT": (255, 255, 80), # Yellow
        "OXIDIZE": (255, 120, 0),  # Orange
        "BIND": (200, 80, 255),    # Purple
    }
    COLOR_TEXT = (220, 220, 240)
    COLOR_UI_BG = (30, 30, 50, 180)
    COLOR_HEALTH_BAR = (80, 255, 80)
    COLOR_HEALTH_BAR_BG = (255, 80, 80)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 20, bold=True)

        self.bg_cells = []
        self._generate_bg_cells()

        # The reset method is called later, but we need an RNG for some initial setup
        # that might happen before the first official reset.
        self.np_random = np.random.default_rng()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_attack_words = random.sample(self.WORD_LIST, k=3)
        self.player_selected_word_idx = 0

        self.attack_cooldown_timer = 0
        self.last_space_state = 0
        self.last_shift_state = 0
        
        self.viruses = []
        self.particles = []
        
        self.virus_spawn_base_rate = 60 # steps
        self.virus_spawn_timer = self.virus_spawn_base_rate

        for _ in range(3):
            self._spawn_virus()

        self.last_dist_to_nearest_virus = self._get_dist_to_nearest_virus()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1
        
        # --- Reward for distance change ---
        current_dist = self._get_dist_to_nearest_virus()
        if self.last_dist_to_nearest_virus is not None and current_dist is not None:
            dist_change = self.last_dist_to_nearest_virus - current_dist
            reward += dist_change * 0.001 # Small continuous reward
        self.last_dist_to_nearest_virus = current_dist

        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Word cycling (on press)
        if shift_held and not self.last_shift_state:
            self.player_selected_word_idx = (self.player_selected_word_idx + 1) % len(self.player_attack_words)
        
        # Attack (on press)
        if space_held and not self.last_space_state and self.attack_cooldown_timer <= 0:
            attack_reward = self._execute_attack()
            reward += attack_reward
            if attack_reward > 0:
                self.attack_cooldown_timer = self.ATTACK_COOLDOWN

        self.last_space_state = space_held
        self.last_shift_state = shift_held

        # --- Update Game Logic ---
        self._update_player(movement)
        self._update_viruses()
        self._update_particles()
        self._update_spawner()
        
        if self.attack_cooldown_timer > 0:
            self.attack_cooldown_timer -= 1

        # --- Termination Check ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated and self.player_health <= 0:
            reward = -100 # Failure
        
        return (
            self._get_observation(),
            np.clip(reward, -100, 100),
            terminated,
            truncated,
            self._get_info(),
        )

    def _get_observation(self):
        self._render_background()
        self._render_particles()
        self._render_viruses()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "viruses_left": len(self.viruses),
        }

    # --- Game Logic Helpers ---

    def _update_player(self, movement):
        velocity = np.zeros(2, dtype=float)
        if movement == 1: velocity[1] -= 1 # Up
        if movement == 2: velocity[1] += 1 # Down
        if movement == 3: velocity[0] -= 1 # Left
        if movement == 4: velocity[0] += 1 # Right

        if np.linalg.norm(velocity) > 0:
            velocity = velocity / np.linalg.norm(velocity) * self.PLAYER_SPEED
        
        self.player_pos += velocity
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.HEIGHT - self.PLAYER_RADIUS)

    def _update_viruses(self):
        for virus in self.viruses:
            dist = np.linalg.norm(self.player_pos - virus['pos'])
            if dist < self.PLAYER_RADIUS + virus['radius']:
                self.player_health -= 1
                self._create_particles(self.player_pos, self.COLOR_HEALTH_BAR_BG, 10)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _update_spawner(self):
        if len(self.viruses) >= self.MAX_VIRUSES:
            return

        self.virus_spawn_timer -= 1
        if self.virus_spawn_timer <= 0:
            self._spawn_virus()
            spawn_rate_decrease = self.steps // 100 * 3
            current_spawn_rate = max(20, self.virus_spawn_base_rate - spawn_rate_decrease)
            self.virus_spawn_timer = int(current_spawn_rate)

    def _spawn_virus(self):
        if len(self.viruses) >= self.MAX_VIRUSES:
            return
        
        weakness = self.np_random.choice(self.WORD_LIST)
        
        for _ in range(100): # Attempt to find a non-overlapping position
            pos = self.np_random.uniform(low=[30, 30], high=[self.WIDTH-30, self.HEIGHT-30])
            
            too_close_to_player = np.linalg.norm(pos - self.player_pos) < 100
            too_close_to_virus = any(np.linalg.norm(pos - v['pos']) < 60 for v in self.viruses)

            if not too_close_to_player and not too_close_to_virus:
                self.viruses.append({
                    'pos': pos,
                    'health': self.VIRUS_MAX_HEALTH,
                    'weakness': weakness,
                    'color': self.COLOR_VIRUS_WEAKNESSES[weakness],
                    'radius': self.VIRUS_RADIUS,
                    'angle_offset': self.np_random.uniform(0, 2 * math.pi)
                })
                return

    def _execute_attack(self):
        reward = 0
        attack_word = self.player_attack_words[self.player_selected_word_idx]
        
        target_virus = None
        min_dist = self.ATTACK_RANGE
        for virus in self.viruses:
            dist = np.linalg.norm(virus['pos'] - self.player_pos)
            if dist < min_dist:
                min_dist = dist
                target_virus = virus
        
        if target_virus:
            if target_virus['weakness'] == attack_word:
                target_virus['health'] -= 1
                reward += 1
                self.score += 10
                self._create_particles(target_virus['pos'], target_virus['color'], 20)
                
                if target_virus['health'] <= 0:
                    self.viruses.remove(target_virus)
                    reward += 10
                    self.score += 100
                    self._create_particles(target_virus['pos'], target_virus['color'], 50, is_explosion=True)
            else:
                self._create_particles(target_virus['pos'], self.COLOR_TEXT, 5, is_explosion=False)
        return reward

    def _check_termination(self):
        if self.player_health <= 0:
            return True
        if not self.viruses: # All viruses eliminated
            return True
        return False

    def _get_dist_to_nearest_virus(self):
        if not self.viruses:
            return None
        distances = [np.linalg.norm(self.player_pos - v['pos']) for v in self.viruses]
        return min(distances)
        
    def _create_particles(self, pos, color, count, is_explosion=False):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4) if is_explosion else self.np_random.uniform(0.5, 2)
            velocity = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': velocity,
                'life': self.np_random.integers(self.PARTICLE_LIFESPAN // 2, self.PARTICLE_LIFESPAN + 1),
                'color': color
            })

    # --- Rendering Helpers ---

    def _generate_bg_cells(self):
        for _ in range(30):
            self.bg_cells.append({
                'pos': np.array([random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT)], dtype=float),
                'vel': np.array([random.uniform(-0.2, 0.2), random.uniform(-0.1, 0.1)]),
                'radius': random.randint(40, 80)
            })

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for cell in self.bg_cells:
            cell['pos'] += cell['vel']
            if cell['pos'][0] < -cell['radius']: cell['pos'][0] = self.WIDTH + cell['radius']
            if cell['pos'][0] > self.WIDTH + cell['radius']: cell['pos'][0] = -cell['radius']
            if cell['pos'][1] < -cell['radius']: cell['pos'][1] = self.HEIGHT + cell['radius']
            if cell['pos'][1] > self.HEIGHT + cell['radius']: cell['pos'][1] = -cell['radius']
            
            pygame.gfxdraw.filled_circle(
                self.screen, int(cell['pos'][0]), int(cell['pos'][1]),
                cell['radius'], self.COLOR_BG_CELLS
            )

    def _render_player(self):
        pos = self.player_pos.astype(int)
        pulse = (math.sin(self.steps * 0.1) + 1) / 2 * 3 # 0 to 3
        radius = self.PLAYER_RADIUS + int(pulse)

        # Glow effect
        for i in range(4):
            glow_radius = radius + i * 4
            alpha = self.COLOR_PLAYER_GLOW[3] * (1 - i / 4)
            temp_color = self.COLOR_PLAYER_GLOW[:3] + (int(alpha),)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, temp_color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], glow_radius, temp_color)

        # Core
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_PLAYER)

    def _render_viruses(self):
        for virus in self.viruses:
            pos = virus['pos'].astype(int)
            points = []
            num_spikes = 7
            rotation = self.steps * 0.02 + virus['angle_offset']
            for i in range(num_spikes * 2):
                r = virus['radius'] if i % 2 == 0 else virus['radius'] * 0.6
                angle = i * math.pi / num_spikes + rotation
                points.append((pos[0] + r * math.cos(angle), pos[1] + r * math.sin(angle)))
            
            pygame.gfxdraw.aapolygon(self.screen, points, virus['color'])
            pygame.gfxdraw.filled_polygon(self.screen, points, virus['color'])

            # Weakness text
            text_surf = self.font_small.render(virus['weakness'], True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(pos[0], pos[1] - virus['radius'] - 10))
            self.screen.blit(text_surf, text_rect)

    def _render_particles(self):
        for p in self.particles:
            pos = p['pos'].astype(int)
            life_ratio = p['life'] / self.PARTICLE_LIFESPAN
            size = max(1, int(life_ratio * 4))
            color = p['color']
            
            if len(color) == 4: # has alpha
                alpha = color[3]
                final_color = color[:3] + (int(alpha * life_ratio),)
            else:
                final_color = (int(color[0]*life_ratio), int(color[1]*life_ratio), int(color[2]*life_ratio))

            pygame.draw.rect(self.screen, final_color, (pos[0], pos[1], size, size))

    def _render_ui(self):
        # Health Bar
        health_ratio = np.clip(self.player_health / self.PLAYER_MAX_HEALTH, 0, 1)
        bar_width = 150
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(bar_width * health_ratio), 20))
        health_text = self.font_small.render(f"HEALTH", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

        # Attack Word Selector
        ui_box_height = 40
        ui_box_rect = pygame.Rect(0, self.HEIGHT - ui_box_height, self.WIDTH, ui_box_height)
        s = pygame.Surface((self.WIDTH, ui_box_height), pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        self.screen.blit(s, (0, self.HEIGHT - ui_box_height))
        
        word = self.player_attack_words[self.player_selected_word_idx]
        word_text = self.font_medium.render(word, True, self.COLOR_VIRUS_WEAKNESSES[word])
        word_rect = word_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT - ui_box_height / 2))
        self.screen.blit(word_text, word_rect)
        
        # Cooldown indicator
        if self.attack_cooldown_timer > 0:
            cooldown_ratio = self.attack_cooldown_timer / self.ATTACK_COOLDOWN
            pygame.draw.line(self.screen, self.COLOR_PLAYER, 
                             (word_rect.left, word_rect.bottom + 2),
                             (word_rect.left + word_rect.width * cooldown_ratio, word_rect.bottom + 2), 2)
        
        # UI prompt
        prompt_text = self.font_small.render("SHIFT to cycle, SPACE to fire", True, self.COLOR_TEXT)
        prompt_rect = prompt_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT - ui_box_height / 2 - 15))
        self.screen.blit(prompt_text, (word_rect.centerx - prompt_rect.width/2, word_rect.top - 14))

    def close(self):
        pygame.quit()

    def _validate_implementation(self):
        # This method is for internal validation and is not part of the standard Gym API
        print("Running implementation validation...")
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
        assert isinstance(reward, (int, float, np.floating))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to run the environment directly for testing.
    # It will use a visible pygame window for rendering.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    env._validate_implementation()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Antibody Assassin")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement_action = 0 # No-op
        space_action = 0
        shift_action = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement_action = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement_action = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement_action = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement_action = 4
        
        if keys[pygame.K_SPACE]: space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_action = 1

        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(30) # Run at 30 FPS

    env.close()