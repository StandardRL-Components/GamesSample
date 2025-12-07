import gymnasium as gym
import os
import pygame
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:22:00.601481
# Source Brief: brief_00424.md
# Brief Index: 424
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Defend a biological cell from invading viruses by synthesizing and deploying matching antibodies."
    user_guide = "Controls: Use ↑↓ to swap antigens and match three of a kind. Use ←→ to aim the portal and press space to deploy the antibody."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.CELL_PADDING = 15
        self.MAX_STEPS = 5000
        self.INITIAL_VIRUSES = 5
        self.MAX_VIRUSES = 30
        self.ANTIGEN_TYPES = 3

        # Colors
        self.COLOR_BG = (10, 20, 30)
        self.COLOR_CELL_WALL = (50, 80, 120)
        self.COLOR_CELL_WALL_GLOW = (80, 120, 180)
        self.COLOR_HEALTH_FG = (40, 220, 150)
        self.COLOR_HEALTH_BG = (180, 40, 80)
        self.COLOR_PORTAL = (0, 191, 255)
        self.COLOR_PORTAL_GLOW = (100, 220, 255)
        self.COLOR_VIRUS = (255, 60, 60)
        self.COLOR_ANTIBODY = (50, 255, 150)
        self.ANTIGEN_COLORS = [(255, 255, 0), (0, 255, 255), (255, 0, 255)]  # Yellow, Cyan, Magenta
        self.COLOR_TEXT = (220, 220, 240)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)

        # --- Game State Initialization ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cellular_health = 0.0
        self.portal_x = 0.0
        self.synthesizer_slots = []
        self.stored_antibody_type = None
        self.viruses = []
        self.antibodies = []
        self.particles = []
        self.virus_base_speed = 0.0
        self.virus_base_health = 0.0
        self.last_action_feedback = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cellular_health = 100.0

        # Entities
        self.viruses = []
        self.antibodies = []
        self.particles = []

        # Player state
        self.portal_x = self.WIDTH / 2
        self.synthesizer_slots = [self.np_random.integers(0, self.ANTIGEN_TYPES) for _ in range(3)]
        self.stored_antibody_type = None

        # Difficulty
        self.virus_base_speed = 0.5
        self.virus_base_health = 1

        # Spawn initial viruses
        for _ in range(self.INITIAL_VIRUSES):
            self._spawn_virus()

        # Visual feedback timers
        self.last_action_feedback = {
            'match_success': 0, 'match_fail': 0, 'deploy': 0, 'cell_damage': 0
        }

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Handle Actions ---
        movement, space_press, _ = action[0], action[1] == 1, action[2] == 1

        if movement in [1, 2]:  # Up/Down: Swap antigens
            if movement == 1:
                self._swap_antigens(0, 1)
            else:
                self._swap_antigens(1, 2)

            match_reward = self._check_antigen_match()
            if match_reward > 0:
                reward += match_reward
                self.last_action_feedback['match_success'] = 15  # frames

        elif movement in [3, 4]:  # Left/Right: Move portal
            portal_speed = 8
            if movement == 3:
                self.portal_x -= portal_speed
            else:
                self.portal_x += portal_speed
            self.portal_x = np.clip(self.portal_x, self.CELL_PADDING + 20, self.WIDTH - self.CELL_PADDING - 20)

        if space_press:
            deploy_reward = self._deploy_antibody()
            if deploy_reward > 0:
                reward += deploy_reward
                self.last_action_feedback['deploy'] = 20  # frames

        # --- Update Game Logic ---
        self._update_difficulty()
        reward += self._update_viruses()
        self._update_antibodies()

        collision_rewards = self._handle_collisions()
        reward += collision_rewards

        self._update_particles()
        self._cleanup_entities()

        # --- Termination Check ---
        terminated = False
        truncated = False
        if self.cellular_health <= 0:
            reward -= 100
            terminated = True
            # sfx: game_over_lose

        if not self.viruses and self.steps > 1:
            reward += 100
            terminated = True
            # sfx: game_over_win

        if self.steps >= self.MAX_STEPS:
            truncated = True

        self.game_over = terminated or truncated
        self.score += reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # --- Game Logic Helpers ---

    def _swap_antigens(self, i, j):
        self.synthesizer_slots[i], self.synthesizer_slots[j] = self.synthesizer_slots[j], self.synthesizer_slots[i]
        # sfx: antigen_swap

    def _check_antigen_match(self):
        if self.stored_antibody_type is None:
            if self.synthesizer_slots[0] == self.synthesizer_slots[1] == self.synthesizer_slots[2]:
                self.stored_antibody_type = self.synthesizer_slots[0]
                self.synthesizer_slots = [self.np_random.integers(0, self.ANTIGEN_TYPES) for _ in range(3)]
                # sfx: match_success
                self._create_particle_burst(
                    (self.WIDTH / 2, self.HEIGHT - 40),
                    self.ANTIGEN_COLORS[self.stored_antibody_type], 20
                )
                return 0.1
        return 0

    def _deploy_antibody(self):
        if self.stored_antibody_type is not None:
            # sfx: deploy_antibody
            self.antibodies.append({
                'pos': np.array([self.portal_x, self.HEIGHT - 100], dtype=float),
                'type': self.stored_antibody_type,
                'target': None,
                'power': 3,  # Can destroy 3 viruses
                'speed': 3.0,
                'radius': 10
            })
            self.stored_antibody_type = None
            return 0.1  # Small reward for deploying
        return 0

    def _update_difficulty(self):
        if self.steps % 200 == 0 and self.steps > 0:
            self.virus_base_speed += 0.05
            self.virus_base_health += 0.05

        if self.steps % 60 == 0 and len(self.viruses) < self.MAX_VIRUSES:
            self._spawn_virus()
            return -0.1  # Penalty for new virus spawn
        return 0

    def _spawn_virus(self):
        pos = np.array([
            self.np_random.uniform(self.CELL_PADDING + 10, self.WIDTH - self.CELL_PADDING - 10),
            self.np_random.uniform(self.CELL_PADDING + 10, self.HEIGHT - 120)
        ])
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.virus_base_speed + self.np_random.uniform(-0.1, 0.1)
        vel = np.array([math.cos(angle), math.sin(angle)]) * speed
        health = int(self.virus_base_health) + (1 if self.np_random.random() < (self.virus_base_health % 1) else 0)

        self.viruses.append({
            'pos': pos, 'vel': vel, 'health': health, 'max_health': health,
            'radius': 8, 'anim_offset': self.np_random.uniform(0, 2 * math.pi)
        })

    def _update_viruses(self):
        reward = 0
        for v in self.viruses:
            v['pos'] += v['vel']
            # Bounce off walls
            if not (self.CELL_PADDING < v['pos'][0] < self.WIDTH - self.CELL_PADDING):
                v['vel'][0] *= -1
                v['pos'][0] = np.clip(v['pos'][0], self.CELL_PADDING, self.WIDTH - self.CELL_PADDING)
                self.cellular_health -= 1
                reward -= 0.5  # Penalty for hitting wall
                self.last_action_feedback['cell_damage'] = 10
                # sfx: cell_damage
            if not (self.CELL_PADDING < v['pos'][1] < self.HEIGHT - self.CELL_PADDING):
                v['vel'][1] *= -1
                v['pos'][1] = np.clip(v['pos'][1], self.CELL_PADDING, self.HEIGHT - self.CELL_PADDING)
                self.cellular_health -= 1
                reward -= 0.5
                self.last_action_feedback['cell_damage'] = 10
                # sfx: cell_damage
        self.cellular_health = max(0, self.cellular_health)
        return reward

    def _update_antibodies(self):
        for ab in self.antibodies:
            # Find new target if needed
            if ab['target'] not in self.viruses or ab['target'] is None:
                ab['target'] = self._find_nearest_virus(ab['pos'])

            # Move towards target
            if ab['target']:
                direction = ab['target']['pos'] - ab['pos']
                dist = np.linalg.norm(direction)
                if dist > 1:
                    ab['pos'] += (direction / dist) * ab['speed']

    def _find_nearest_virus(self, pos):
        nearest_virus = None
        min_dist_sq = float('inf')
        for v in self.viruses:
            dist_sq = np.sum((v['pos'] - pos) ** 2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest_virus = v
        return nearest_virus

    def _handle_collisions(self):
        reward = 0
        for ab in self.antibodies:
            for v in self.viruses:
                dist = np.linalg.norm(ab['pos'] - v['pos'])
                if dist < ab['radius'] + v['radius']:
                    # sfx: virus_hit
                    self._create_particle_burst(v['pos'], self.COLOR_VIRUS, 10)
                    v['health'] -= 1
                    ab['power'] -= 1
                    reward += 0.01  # Reward for damaging
                    if v['health'] <= 0:
                        reward += 1  # Reward for destroying
                        # sfx: virus_destroy
                        self._create_particle_burst(v['pos'], self.COLOR_VIRUS, 30, True)
                    break  # Antibody hits one virus per frame
        return reward

    def _cleanup_entities(self):
        self.viruses = [v for v in self.viruses if v['health'] > 0]
        self.antibodies = [ab for ab in self.antibodies if ab['power'] > 0]
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _create_particle_burst(self, pos, color, count, explosion=False):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4) if explosion else self.np_random.uniform(0.5, 2)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            lifespan = self.np_random.integers(15, 30) if explosion else self.np_random.integers(10, 20)
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'color': color,
                'lifespan': lifespan, 'max_lifespan': lifespan
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95  # Drag
            p['lifespan'] -= 1

    # --- Rendering ---

    def _get_observation(self):
        self._render_background()
        self._render_particles()
        self._render_viruses()
        self._render_antibodies()
        self._render_portal()
        self._render_synthesizer()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)

        # Cell wall effect
        flash_alpha = int(90 * (self.last_action_feedback['cell_damage'] / 10))
        if flash_alpha > 0:
            pygame.gfxdraw.box(self.screen, self.screen.get_rect(), (*self.COLOR_VIRUS, flash_alpha))
            self.last_action_feedback['cell_damage'] -= 1

        # Draw cell wall with glow
        rect = pygame.Rect(self.CELL_PADDING, self.CELL_PADDING,
                           self.WIDTH - 2 * self.CELL_PADDING, self.HEIGHT - 2 * self.CELL_PADDING)
        for i in range(5, 0, -1):
            glow_color = (*self.COLOR_CELL_WALL_GLOW, 20 - i * 3)
            pygame.gfxdraw.rectangle(self.screen, rect.inflate(i * 2, i * 2), glow_color)
        pygame.draw.rect(self.screen, self.COLOR_CELL_WALL, rect, 2, 5)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = (*p['color'], alpha)
            size = int(3 * (p['lifespan'] / p['max_lifespan']))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), size, color)

    def _render_viruses(self):
        for v in self.viruses:
            pos = (int(v['pos'][0]), int(v['pos'][1]))
            radius = v['radius']

            # Pulsating spiky shape
            points = []
            num_points = 7
            for i in range(num_points):
                angle = 2 * math.pi * i / num_points + v['anim_offset'] + (self.steps / 30.0)
                r = radius + math.sin(angle * 3 + self.steps / 10.0) * (radius * 0.3)
                points.append((pos[0] + r * math.cos(angle), pos[1] + r * math.sin(angle)))

            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_VIRUS)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_VIRUS)

            # Health bar
            if v['health'] < v['max_health']:
                bar_width = 15
                bar_height = 3
                bar_x = pos[0] - bar_width / 2
                bar_y = pos[1] - radius - 8
                health_pct = v['health'] / v['max_health']
                pygame.draw.rect(self.screen, (80, 0, 0), (bar_x, bar_y, bar_width, bar_height))
                pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (bar_x, bar_y, bar_width * health_pct, bar_height))

    def _render_antibodies(self):
        for ab in self.antibodies:
            pos = (int(ab['pos'][0]), int(ab['pos'][1]))
            radius = int(ab['radius'] * (ab['power'] / 3.0 * 0.5 + 0.5))

            # Glow effect
            glow_radius = int(radius * 1.8)
            glow_alpha = 80
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, (*self.COLOR_ANTIBODY, glow_alpha))

            # Main shape
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_ANTIBODY)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ANTIBODY)

    def _render_portal(self):
        pos_y = self.HEIGHT - 100
        pos = (int(self.portal_x), pos_y)

        # Deploy flash effect
        flash_progress = self.last_action_feedback['deploy'] / 20.0
        if flash_progress > 0:
            flash_radius = int(20 + 80 * (1 - flash_progress))
            flash_alpha = int(200 * flash_progress)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], flash_radius,
                                          (*self.COLOR_PORTAL_GLOW, flash_alpha))
            self.last_action_feedback['deploy'] -= 1

        # Animated portal
        for i in range(1, 6):
            radius = 15 + int(math.sin(self.steps / 10.0 + i) * 3)
            alpha = 150 - i * 20
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (*self.COLOR_PORTAL, alpha))

        # Central glow
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, (*self.COLOR_PORTAL_GLOW, 100))

    def _render_synthesizer(self):
        base_y = self.HEIGHT - 50
        slot_size = 24
        spacing = 40
        total_width = 3 * slot_size + 2 * spacing
        start_x = (self.WIDTH - total_width) / 2 + slot_size / 2

        # Match success/fail flash
        if self.last_action_feedback['match_success'] > 0:
            flash_color = (*self.COLOR_ANTIBODY, int(150 * self.last_action_feedback['match_success'] / 15.0))
            pygame.draw.rect(self.screen, flash_color, (start_x - 50, base_y - 25, 180, 50), 0, 10)
            self.last_action_feedback['match_success'] -= 1

        # Slots and antigens
        for i, antigen_type in enumerate(self.synthesizer_slots):
            pos_x = start_x + i * (slot_size + spacing)

            # Draw slot
            pygame.draw.rect(self.screen, self.COLOR_CELL_WALL,
                             (pos_x - slot_size / 2, base_y - slot_size / 2, slot_size, slot_size), 1, 3)

            # Draw antigen
            color = self.ANTIGEN_COLORS[antigen_type]
            if antigen_type == 0:  # Circle
                pygame.gfxdraw.aacircle(self.screen, int(pos_x), int(base_y), 8, color)
                pygame.gfxdraw.filled_circle(self.screen, int(pos_x), int(base_y), 8, color)
            elif antigen_type == 1:  # Square
                size = 16
                rect = pygame.Rect(pos_x - size / 2, base_y - size / 2, size, size)
                pygame.draw.rect(self.screen, color, rect, 0, 2)
            elif antigen_type == 2:  # Triangle
                points = [
                    (pos_x, base_y - 9),
                    (pos_x - 9, base_y + 6),
                    (pos_x + 9, base_y + 6),
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)

        # Stored antibody indicator
        if self.stored_antibody_type is not None:
            pos_x = self.WIDTH / 2
            pos_y = self.HEIGHT - 85
            color = self.ANTIGEN_COLORS[self.stored_antibody_type]
            pygame.gfxdraw.filled_circle(self.screen, int(pos_x), int(pos_y), 10, (*color, 100))
            pygame.gfxdraw.aacircle(self.screen, int(pos_x), int(pos_y), 10, color)
            text = self.font_small.render("READY", True, self.COLOR_TEXT)
            self.screen.blit(text, text.get_rect(center=(pos_x, pos_y - 20)))

    def _render_ui(self):
        # Health Bar
        health_rect = pygame.Rect(25, 25, 200, 20)
        health_pct = self.cellular_health / 100.0
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, health_rect, 0, 5)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG,
                         (health_rect.x, health_rect.y, health_rect.width * health_pct, health_rect.height), 0, 5)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, health_rect, 1, 5)

        # Virus Count
        virus_text = self.font_main.render(f"VIRUSES: {len(self.viruses)}", True, self.COLOR_TEXT)
        self.screen.blit(virus_text, virus_text.get_rect(topright=(self.WIDTH - 25, 22)))

        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, score_text.get_rect(topright=(self.WIDTH - 25, 50)))

        # Game Over
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            message = "VICTORY" if self.cellular_health > 0 and not self.viruses else "CELLULAR COLLAPSE"
            end_text = pygame.font.Font(None, 60).render(message, True, self.COLOR_TEXT)
            self.screen.blit(end_text, end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 20)))

            final_score_text = self.font_main.render(f"Final Score: {int(self.score)}", True, self.COLOR_TEXT)
            self.screen.blit(final_score_text,
                             final_score_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 + 30)))

    # --- Gymnasium Interface ---

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cellular_health": self.cellular_health,
            "virus_count": len(self.viruses),
            "stored_antibody": self.stored_antibody_type is not None
        }

    def close(self):
        pygame.quit()


# --- Example Usage ---
if __name__ == "__main__":
    # The original code had a validation method that is not part of the Gym API
    # and would fail in a headless environment if not removed/adapted.
    # It has been removed. The following is for human play.
    
    # Re-enable video driver for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" 

    env = GameEnv(render_mode="rgb_array")

    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Cellular Defense")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    
    # Game loop for human play
    while not done:
        # Action mapping for human keyboard
        keys = pygame.key.get_pressed()
        movement = 0  # no-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # Process Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30)  # Run at 30 FPS

    env.close()