import gymnasium as gym
import os
import pygame
import numpy as np
import math
from gymnasium.spaces import MultiDiscrete, Box
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame.init()
pygame.font.init()


class GameEnv(gym.Env):
    """
    Gymnasium environment: Defend a neon base from rhythmic waves of color-coded enemies
    by strategically cloning yourself to match their patterns and counter-attack.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Defend your neon base from waves of color-coded enemies by deploying matching "
        "defensive clones and ordering them to attack."
    )
    user_guide = (
        "Controls: Use arrow keys to select weapon color (↑ Red, ↓ Green, ← Blue, → Cycle). "
        "Press Space to deploy a clone and Shift to fire."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 2000

        # Colors (Neon-Cyberpunk Theme)
        self.COLOR_BG = (10, 5, 25)
        self.COLOR_UI_TEXT = (220, 220, 255)
        self.COLOR_RED = (255, 20, 50)
        self.COLOR_GREEN = (20, 255, 100)
        self.COLOR_BLUE = (20, 100, 255)
        self.COLORS = [self.COLOR_RED, self.COLOR_GREEN, self.COLOR_BLUE]
        self.COLOR_NAMES = ["RED", "GREEN", "BLUE"]

        # Game Parameters
        self.BASE_POS = (self.WIDTH // 2, self.HEIGHT // 2)
        self.BASE_RADIUS = 30
        self.INITIAL_BASE_HEALTH = 100
        self.BASE_HIT_DAMAGE = 25
        
        self.CLONE_RADIUS = 12
        self.CLONE_ORBIT_RADIUS = 80
        self.NUM_CLONE_SLOTS = 6
        self.CLONE_SLOTS = self._calculate_clone_slots()

        self.ENEMY_RADIUS = 10
        self.ENEMY_SPAWN_RADIUS = min(self.WIDTH, self.HEIGHT) / 2 + 50
        self.ENEMY_SPEED = 1.5
        self.ENEMY_HEALTH = 100
        self.MATCHED_ATTACK_DAMAGE = 100
        self.MISMATCHED_ATTACK_DAMAGE = 34 # 3 hits to kill
        
        self.BEAM_SPEED = 12
        self.BEAM_WIDTH = 5
        
        self.ENEMY_SPAWN_RATE_INITIAL = 1.0 / (self.FPS * 1.5) # 1 enemy every 1.5s
        self.ENEMY_SPAWN_RATE_INCREASE = 0.00005

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.clones = []
        self.enemies = []
        self.beams = []
        self.particles = []
        self.current_color_idx = 0
        self.current_clone_slot = 0
        self.last_action = np.array([0, 0, 0])
        self.enemy_spawn_timer = 0.0
        self.enemy_spawn_rate = 0.0
        self.reward_this_step = 0
        self.screen_shake = 0
        self.available_colors = [0] # Start with only Red

    def _calculate_clone_slots(self):
        slots = []
        for i in range(self.NUM_CLONE_SLOTS):
            angle = 2 * math.pi * i / self.NUM_CLONE_SLOTS
            x = self.BASE_POS[0] + self.CLONE_ORBIT_RADIUS * math.cos(angle)
            y = self.BASE_POS[1] + self.CLONE_ORBIT_RADIUS * math.sin(angle)
            slots.append((int(x), int(y)))
        return slots

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = self.INITIAL_BASE_HEALTH
        
        self.clones = []
        self.enemies = []
        self.beams = []
        self.particles = []
        
        self.current_color_idx = 0
        self.current_clone_slot = 0
        self.last_action = np.array([0, 0, 0])
        
        self.enemy_spawn_timer = 0.0
        self.enemy_spawn_rate = self.ENEMY_SPAWN_RATE_INITIAL
        self.available_colors = [0] # Start with only Red
        
        self.reward_this_step = 0
        self.screen_shake = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.reward_this_step = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self._handle_input(action)
        self._update_game_state()
        
        self.steps += 1
        
        reward = 0.1  # Survival reward
        reward += self.reward_this_step

        terminated = self.base_health <= 0
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated or truncated:
            self.game_over = True
            if self.base_health > 0 and truncated:
                reward += 100 # Win bonus
                self.score += 10000 # Visual score bonus for winning
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        # Unpack factorized action
        movement_key = action[0]
        space_pressed = action[1] == 1 and self.last_action[1] == 0
        shift_pressed = action[2] == 1 and self.last_action[2] == 0

        # Color selection (on key press)
        if movement_key != self.last_action[0] and movement_key != 0:
            if movement_key == 1 and 0 in self.available_colors: self.current_color_idx = 0
            elif movement_key == 2 and 1 in self.available_colors: self.current_color_idx = 1
            elif movement_key == 3 and 2 in self.available_colors: self.current_color_idx = 2
            elif movement_key == 4: # Cycle
                if self.current_color_idx in self.available_colors:
                    current_selection_index = self.available_colors.index(self.current_color_idx)
                    self.current_color_idx = self.available_colors[(current_selection_index + 1) % len(self.available_colors)]
                else: # Fallback if current color is somehow not available
                    self.current_color_idx = self.available_colors[0]
                
        # Deploy clone
        if space_pressed:
            self._deploy_clone()
        
        # Trigger attack
        if shift_pressed:
            self._trigger_attack()

        self.last_action = action

    def _deploy_clone(self):
        slot_pos = self.CLONE_SLOTS[self.current_clone_slot]
        
        # Overwrite if a clone is already in the slot
        existing_clone = next((c for c in self.clones if c['pos'] == slot_pos), None)
        if existing_clone:
            existing_clone['color_idx'] = self.current_color_idx
            existing_clone['spawn_timer'] = self.FPS // 4 # re-spawn animation
        else:
            self.clones.append({
                'pos': slot_pos,
                'color_idx': self.current_color_idx,
                'spawn_timer': self.FPS // 4 # 0.25s animation
            })
        
        self._create_particles(slot_pos, self.COLORS[self.current_color_idx], 20, 5)
        self.current_clone_slot = (self.current_clone_slot + 1) % self.NUM_CLONE_SLOTS

    def _trigger_attack(self):
        for clone in self.clones:
            # Find nearest enemy
            if not self.enemies: continue
            
            nearest_enemy = min(self.enemies, key=lambda e: math.hypot(e['pos'][0] - clone['pos'][0], e['pos'][1] - clone['pos'][1]))
            
            target_pos = nearest_enemy['pos']
            start_pos = clone['pos']
            
            angle = math.atan2(target_pos[1] - start_pos[1], target_pos[0] - start_pos[0])
            vel = (self.BEAM_SPEED * math.cos(angle), self.BEAM_SPEED * math.sin(angle))
            
            self.beams.append({
                'pos': list(start_pos),
                'vel': vel,
                'color_idx': clone['color_idx']
            })

    def _update_game_state(self):
        # Update progression (difficulty)
        if self.steps > 0 and self.steps % 100 == 0:
            self.enemy_spawn_rate += self.ENEMY_SPAWN_RATE_INCREASE
        if self.steps == 500 and 1 not in self.available_colors: self.available_colors.append(1) # Unlock Green
        if self.steps == 1000 and 2 not in self.available_colors: self.available_colors.append(2) # Unlock Blue
            
        # Spawn enemies
        self.enemy_spawn_timer += self.enemy_spawn_rate
        if self.enemy_spawn_timer >= 1.0:
            self.enemy_spawn_timer -= 1.0
            self._spawn_enemy()

        # Update clones (spawn animation)
        for c in self.clones:
            if c['spawn_timer'] > 0:
                c['spawn_timer'] -= 1

        # Update beams
        for beam in self.beams[:]:
            beam['pos'][0] += beam['vel'][0]
            beam['pos'][1] += beam['vel'][1]
            
            # Check for collision with enemies
            hit_enemy = False
            for enemy in self.enemies[:]:
                if math.hypot(beam['pos'][0] - enemy['pos'][0], beam['pos'][1] - enemy['pos'][1]) < self.ENEMY_RADIUS:
                    damage = self.MATCHED_ATTACK_DAMAGE if beam['color_idx'] == enemy['color_idx'] else self.MISMATCHED_ATTACK_DAMAGE
                    enemy['health'] -= damage
                    self._create_particles(enemy['pos'], self.COLORS[beam['color_idx']], 15, 3)
                    
                    if enemy['health'] <= 0:
                        self.enemies.remove(enemy)
                        self.score += 100
                        self.reward_this_step += 1.0
                        self._create_particles(enemy['pos'], self.COLORS[enemy['color_idx']], 40, 8)
                    
                    hit_enemy = True
                    break
            
            if hit_enemy or not (0 < beam['pos'][0] < self.WIDTH and 0 < beam['pos'][1] < self.HEIGHT):
                if beam in self.beams:
                    self.beams.remove(beam)

        # Update enemies
        for enemy in self.enemies[:]:
            # Check for blocking by clones
            is_blocked = False
            for clone in self.clones:
                dist_to_clone = math.hypot(enemy['pos'][0] - clone['pos'][0], enemy['pos'][1] - clone['pos'][1])
                if dist_to_clone < self.ENEMY_RADIUS + self.CLONE_RADIUS and clone['color_idx'] == enemy['color_idx']:
                    is_blocked = True
                    self._create_particles(enemy['pos'], self.COLORS[clone['color_idx']], 5, 1, 0.5)
                    break
            
            # Move enemy if not blocked
            if not is_blocked:
                angle = math.atan2(self.BASE_POS[1] - enemy['pos'][1], self.BASE_POS[0] - enemy['pos'][0])
                enemy['pos'] = (enemy['pos'][0] + self.ENEMY_SPEED * math.cos(angle), 
                                enemy['pos'][1] + self.ENEMY_SPEED * math.sin(angle))

            # Check for collision with base
            if math.hypot(enemy['pos'][0] - self.BASE_POS[0], enemy['pos'][1] - self.BASE_POS[1]) < self.BASE_RADIUS:
                self.enemies.remove(enemy)
                self.base_health -= self.BASE_HIT_DAMAGE
                self.score -= 500
                self.reward_this_step -= 5.0
                self.screen_shake = 15
                self._create_particles(self.BASE_POS, (255, 255, 255), 50, 10)

        # Update particles
        for p in self.particles[:]:
            p['lifespan'] -= 1
            p['radius'] *= 0.95
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _spawn_enemy(self):
        angle = self.np_random.uniform(0, 2 * math.pi)
        x = self.BASE_POS[0] + self.ENEMY_SPAWN_RADIUS * math.cos(angle)
        y = self.BASE_POS[1] + self.ENEMY_SPAWN_RADIUS * math.sin(angle)
        color_idx = self.np_random.choice(self.available_colors)
        
        self.enemies.append({
            'pos': (x, y),
            'color_idx': color_idx,
            'health': self.ENEMY_HEALTH
        })

    def _create_particles(self, pos, color, count, max_radius, max_lifespan_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            radius = self.np_random.uniform(1, max_radius)
            offset_x = radius * math.cos(angle)
            offset_y = radius * math.sin(angle)
            
            self.particles.append({
                'pos': (pos[0] + offset_x, pos[1] + offset_y),
                'color': color,
                'radius': self.np_random.uniform(1, 4),
                'lifespan': self.np_random.integers(10, int(20 * max_lifespan_mult) + 1),
            })

    def _get_observation(self):
        # Apply screen shake
        render_offset = (0, 0)
        if self.screen_shake > 0:
            self.screen_shake -= 1
            render_offset = (self.np_random.integers(-self.screen_shake, self.screen_shake + 1),
                             self.np_random.integers(-self.screen_shake, self.screen_shake + 1))

        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game(render_offset)
        
        # Render UI overlay (not affected by shake)
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _render_game(self, offset):
        # Render particles (background layer)
        for p in self.particles:
            pos = (int(p['pos'][0] + offset[0]), int(p['pos'][1] + offset[1]))
            pygame.draw.circle(self.screen, p['color'], pos, max(0, int(p['radius'])))

        # Render base
        self._draw_glowing_circle(self.screen, (100, 100, 255), (self.BASE_POS[0] + offset[0], self.BASE_POS[1] + offset[1]), self.BASE_RADIUS, 15)

        # Render clones
        for c in self.clones:
            radius_mult = 1.0
            if c['spawn_timer'] > 0:
                progress = 1.0 - (c['spawn_timer'] / (self.FPS // 4))
                radius_mult = 0.5 + 0.5 * progress # Grow in
            pos = (int(c['pos'][0] + offset[0]), int(c['pos'][1] + offset[1]))
            self._draw_glowing_circle(self.screen, self.COLORS[c['color_idx']], pos, int(self.CLONE_RADIUS * radius_mult), 10)

        # Render beams
        for b in self.beams:
            pos = (int(b['pos'][0] + offset[0]), int(b['pos'][1] + offset[1]))
            end_pos = (int(pos[0] - b['vel'][0] * 0.5), int(pos[1] - b['vel'][1] * 0.5))
            color = self.COLORS[b['color_idx']]
            glow_color = (*color, 60)
            # Glow line
            pygame.draw.line(self.screen, glow_color, pos, end_pos, self.BEAM_WIDTH * 3)
            # Core line
            pygame.draw.line(self.screen, color, pos, end_pos, self.BEAM_WIDTH)

        # Render enemies
        for e in self.enemies:
            pos = (int(e['pos'][0] + offset[0]), int(e['pos'][1] + offset[1]))
            self._draw_glowing_poly(self.screen, self.COLORS[e['color_idx']], pos, self.ENEMY_RADIUS, 3, 10)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Steps / Timer
        steps_text = self.font_large.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

        # Base Health Bar
        health_ratio = max(0, self.base_health / self.INITIAL_BASE_HEALTH)
        health_bar_width = 200
        health_bar_height = 15
        health_bar_pos = ((self.WIDTH - health_bar_width) // 2, 15)
        pygame.draw.rect(self.screen, (50, 50, 80), (*health_bar_pos, health_bar_width, health_bar_height), border_radius=4)
        health_color = (255, 50, 50) if health_ratio < 0.3 else (50, 200, 255)
        pygame.draw.rect(self.screen, health_color, (*health_bar_pos, int(health_bar_width * health_ratio), health_bar_height), border_radius=4)

        # Current Selection Preview
        preview_pos = (self.WIDTH // 2, self.HEIGHT - 40)
        preview_text = self.font_small.render("WEAPON:", True, self.COLOR_UI_TEXT)
        self.screen.blit(preview_text, (preview_pos[0] - 80, preview_pos[1] - 8))
        
        color_name = self.COLOR_NAMES[self.current_color_idx]
        color_text = self.font_large.render(color_name, True, self.COLORS[self.current_color_idx])
        self.screen.blit(color_text, (preview_pos[0] - 20, preview_pos[1] - 14))

        # Next Clone Slot Indicator
        if not self.game_over:
            slot_pos = self.CLONE_SLOTS[self.current_clone_slot]
            self._draw_hollow_circle(self.screen, self.COLORS[self.current_color_idx], slot_pos, self.CLONE_RADIUS + 4, 2)

    def _draw_glowing_circle(self, surface, color, center, radius, glow_width):
        if radius <= 0: return
        center = (int(center[0]), int(center[1]))
        
        # Outer glow layers
        for i in range(glow_width, 0, -2):
            alpha = int(90 * (1 - i / glow_width))
            glow_color = (*color, alpha)
            # Use a temporary surface for blending
            temp_surf = pygame.Surface((radius * 2 + i * 2, radius * 2 + i * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, glow_color, (radius + i, radius + i), radius + i)
            surface.blit(temp_surf, (center[0] - radius - i, center[1] - radius - i), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Core circle
        pygame.draw.circle(surface, color, center, radius)

    def _draw_glowing_poly(self, surface, color, center, radius, sides, glow_width):
        if radius <= 0: return
        
        # Calculate points for a regular polygon
        def get_poly_points(c, r, n, rot=0):
            return [(c[0] + r * math.cos(rot + 2 * math.pi * i / n), c[1] + r * math.sin(rot + 2 * math.pi * i / n)) for i in range(n)]

        angle_to_base = math.atan2(self.BASE_POS[1] - center[1], self.BASE_POS[0] - center[0])
        points = get_poly_points(center, radius, sides, angle_to_base)
        
        # Outer glow
        for i in range(glow_width, 0, -2):
            alpha = int(90 * (1 - i / glow_width))
            glow_color = (*color, alpha)
            glow_points = get_poly_points(center, radius + i, sides, angle_to_base)
            pygame.draw.polygon(surface, glow_color, glow_points)

        # Core polygon
        pygame.draw.polygon(surface, color, points)

    def _draw_hollow_circle(self, surface, color, center, radius, width):
        if radius <= 0: return
        center = (int(center[0]), int(center[1]))
        pygame.draw.circle(surface, color, center, radius, width)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "enemies": len(self.enemies),
            "clones": len(self.clones)
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Neon Base Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # no-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        # Use a more intuitive mapping for manual play
        if keys[pygame.K_1]: movement = 1 # Red
        elif keys[pygame.K_2]: movement = 2 # Green
        elif keys[pygame.K_3]: movement = 3 # Blue
        elif keys[pygame.K_c]: movement = 4 # Cycle
        
        # The test environment uses arrow keys, so we can support both
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = np.array([movement, space, shift])
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Transpose the observation back for Pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            running = False # End after one game
            
        clock.tick(env.FPS)
        
    env.close()