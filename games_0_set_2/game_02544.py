import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Press space to fire a projectile at the nearest enemy."
    )

    game_description = (
        "Defeat waves of procedurally generated monsters in a top-down, neon-infused arena to achieve the highest score."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 10000
        self.ARENA_RADIUS = 180

        # Colors
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_ARENA = (20, 10, 40)
        self.COLOR_ARENA_BORDER = (80, 50, 120)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_GLOW = (0, 255, 150, 50)
        self.COLOR_PROJECTILE = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 255)
        self.MONSTER_COLORS = {
            'direct': (255, 50, 50),
            'zigzag': (255, 150, 50),
            'circular': (200, 50, 255)
        }
        self.HEALTH_BAR_BG = (50, 50, 50)
        self.HEALTH_BAR_FG = (200, 0, 0)

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Game state variables (initialized in reset)
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_health = None
        self.player_max_health = 3
        self.player_speed = 4
        self.player_radius = 10
        self.monsters = []
        self.projectiles = []
        self.particles = []
        self.wave = 0
        self.monsters_per_wave = 15
        self.base_monster_speed = 0.5
        self.shoot_cooldown = 0
        self.max_shoot_cooldown = 8 # frames
        self.prev_space_held = False
        self.screen_shake = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.wave = 0
        self.game_over = False

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_health = self.player_max_health
        
        self.monsters.clear()
        self.projectiles.clear()
        self.particles.clear()
        
        self.shoot_cooldown = 0
        self.prev_space_held = False
        self.screen_shake = 0
        
        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        # Unpack action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Detect rising edge for shooting
        space_pressed_this_frame = space_held and not self.prev_space_held
        self.prev_space_held = space_held

        if not self.game_over:
            # Update cooldowns and effects
            if self.shoot_cooldown > 0: self.shoot_cooldown -= 1
            if self.screen_shake > 0: self.screen_shake -= 1

            # Player actions
            self._handle_player_movement(movement)
            if space_pressed_this_frame and self.shoot_cooldown == 0:
                reward += self._handle_shooting()

            # Update game entities
            self._update_monsters()
            reward += self._update_projectiles()
            self._update_particles()
            
            # Handle collisions and game events
            reward += self._handle_collisions()

            # Check for wave clear
            if not self.monsters:
                reward += 10  # Wave clear bonus
                self._spawn_wave()
                self._create_particle_burst(self.player_pos, 100, self.COLOR_PLAYER, 5, 80)

        self.steps += 1
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over:
            self.game_over = True
        
        if self.game_over and self.player_health <= 0:
            reward -= 100 # Large penalty for dying

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        # Create a temporary surface for screen shake
        render_surface = self.screen.copy()
        render_surface.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_arena(render_surface)
        self._render_particles(render_surface)
        self._render_projectiles(render_surface)
        self._render_monsters(render_surface)
        self._render_player(render_surface)
        
        # Apply screen shake
        shake_offset = (0, 0)
        if self.screen_shake > 0:
            shake_offset = (self.np_random.integers(-5, 6), self.np_random.integers(-5, 6))
        self.screen.blit(render_surface, shake_offset)
        
        # Render UI overlay (not affected by shake)
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "health": self.player_health,
        }

    def _check_termination(self):
        if self.player_health <= 0 and not self.game_over:
            self.game_over = True
            self._create_particle_burst(self.player_pos, 200, self.COLOR_PLAYER, 8, 100)
        return self.game_over

    def _spawn_wave(self):
        self.wave += 1
        monster_speed = self.base_monster_speed + (self.wave - 1) * 0.05
        
        for _ in range(self.monsters_per_wave):
            # Spawn on the edge of the screen
            angle = self.np_random.uniform(0, 2 * math.pi)
            spawn_dist = self.np_random.uniform(self.ARENA_RADIUS + 20, self.ARENA_RADIUS + 50)
            center = np.array([self.WIDTH / 2, self.HEIGHT / 2])
            pos = center + np.array([math.cos(angle) * spawn_dist, math.sin(angle) * spawn_dist])
            
            pattern = self.np_random.choice(['direct', 'zigzag', 'circular'])
            
            self.monsters.append({
                'pos': pos.astype(np.float32),
                'radius': self.np_random.integers(6, 10),
                'health': self.np_random.integers(1, 4),
                'max_health': 3,
                'speed': monster_speed * self.np_random.uniform(0.8, 1.2),
                'pattern': pattern,
                'color': self.MONSTER_COLORS[pattern],
                'zigzag_timer': 0,
                'zigzag_dir': 1
            })

    def _handle_player_movement(self, movement):
        velocity = np.zeros(2, dtype=np.float32)
        if movement == 1: velocity[1] = -1  # Up
        if movement == 2: velocity[1] = 1   # Down
        if movement == 3: velocity[0] = -1  # Left
        if movement == 4: velocity[0] = 1   # Right
        
        if np.linalg.norm(velocity) > 0:
            velocity = velocity / np.linalg.norm(velocity)
        
        self.player_pos += velocity * self.player_speed
        
        # Arena collision
        center = np.array([self.WIDTH / 2, self.HEIGHT / 2])
        dist_to_center = np.linalg.norm(self.player_pos - center)
        if dist_to_center > self.ARENA_RADIUS - self.player_radius:
            direction = (self.player_pos - center) / dist_to_center
            self.player_pos = center + direction * (self.ARENA_RADIUS - self.player_radius)

    def _handle_shooting(self):
        self.shoot_cooldown = self.max_shoot_cooldown
        nearest_monster = self._find_nearest_monster()
        
        if nearest_monster is None:
            return 0 # No monsters to shoot at
        
        direction = nearest_monster['pos'] - self.player_pos
        dist = np.linalg.norm(direction)
        if dist > 0:
            direction /= dist
        else: # Monster is on top of player
            direction = np.array([1, 0], dtype=np.float32)

        self.projectiles.append({
            'pos': self.player_pos.copy(),
            'vel': direction * 12,
            'radius': 3,
            'lifespan': 60 # frames
        })
        return 0

    def _find_nearest_monster(self):
        if not self.monsters:
            return None
        
        min_dist_sq = float('inf')
        nearest = None
        for monster in self.monsters:
            dist_sq = np.sum((monster['pos'] - self.player_pos)**2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest = monster
        return nearest

    def _update_monsters(self):
        for m in self.monsters:
            direction = self.player_pos - m['pos']
            dist = np.linalg.norm(direction)
            
            if dist > 1: # Avoid division by zero
                direction /= dist
            
            if m['pattern'] == 'direct':
                m['pos'] += direction * m['speed']
            elif m['pattern'] == 'zigzag':
                m['zigzag_timer'] += 1
                if m['zigzag_timer'] > 30:
                    m['zigzag_timer'] = 0
                    m['zigzag_dir'] *= -1
                perp_dir = np.array([-direction[1], direction[0]])
                m['pos'] += (direction + perp_dir * m['zigzag_dir'] * 0.7) * m['speed']
            elif m['pattern'] == 'circular':
                tangent_dir = np.array([-direction[1], direction[0]])
                orbit_dist = 100
                if dist > orbit_dist + 5: # Move towards orbit
                    m['pos'] += direction * m['speed']
                elif dist < orbit_dist - 5: # Move away from orbit
                    m['pos'] -= direction * m['speed']
                else: # Orbit
                    m['pos'] += tangent_dir * m['speed']

    def _update_projectiles(self):
        reward = 0
        projectiles_to_keep = []
        for p in self.projectiles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            
            center = np.array([self.WIDTH / 2, self.HEIGHT / 2])
            dist_to_center = np.linalg.norm(p['pos'] - center)
            
            if p['lifespan'] > 0 and dist_to_center < self.ARENA_RADIUS + 10:
                projectiles_to_keep.append(p)
            else:
                reward -= 0.01 # Penalty for missing
        self.projectiles = projectiles_to_keep
        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] = max(0, p['radius'] - 0.1)

    def _handle_collisions(self):
        reward = 0
        
        # Projectiles vs Monsters
        remaining_projectiles = []
        for p in self.projectiles:
            hit = False
            for m in self.monsters:
                dist_sq = np.sum((p['pos'] - m['pos'])**2)
                if dist_sq < (p['radius'] + m['radius'])**2:
                    m['health'] -= 1
                    hit = True
                    reward += 0.1 # Hit reward
                    self._create_particle_burst(p['pos'], 10, m['color'], 2, 20)
                    break
            if not hit:
                remaining_projectiles.append(p)
        self.projectiles = remaining_projectiles
        
        # Check for defeated monsters
        monsters_alive = []
        for m in self.monsters:
            if m['health'] > 0:
                monsters_alive.append(m)
            else:
                self.score += 10
                reward += 1 # Defeat reward
                self._create_particle_burst(m['pos'], 50, m['color'], 4, 40)
        self.monsters = monsters_alive
        
        # Player vs Monsters
        for m in self.monsters:
            dist_sq = np.sum((self.player_pos - m['pos'])**2)
            if dist_sq < (self.player_radius + m['radius'])**2:
                self.player_health -= 1
                self.screen_shake = 10
                self._create_particle_burst(self.player_pos, 30, (255, 0, 0), 3, 30)
                # Move monster away slightly to prevent instant multi-hits
                direction = m['pos'] - self.player_pos
                if np.linalg.norm(direction) > 0:
                    m['pos'] += direction / np.linalg.norm(direction) * (self.player_radius + m['radius'])
                break # Only take one hit per frame
                
        return reward

    def _create_particle_burst(self, pos, count, color, max_speed, max_life):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed) / 10.0
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            lifespan = self.np_random.integers(max_life // 2, max_life)
            self.particles.append({
                'pos': pos.copy() + vel * 2,
                'vel': vel,
                'lifespan': lifespan,
                'max_lifespan': lifespan,  # Store initial lifespan for fading calculation
                'color': color,
                'radius': self.np_random.uniform(1, 3)
            })

    def _render_arena(self, surface):
        center_pos = (self.WIDTH // 2, self.HEIGHT // 2)
        pygame.gfxdraw.filled_circle(surface, center_pos[0], center_pos[1], self.ARENA_RADIUS, self.COLOR_ARENA)
        pygame.gfxdraw.aacircle(surface, center_pos[0], center_pos[1], self.ARENA_RADIUS, self.COLOR_ARENA_BORDER)

    def _render_player(self, surface):
        if self.player_health > 0:
            pos_int = (int(self.player_pos[0]), int(self.player_pos[1]))
            # Glow effect
            glow_surf = pygame.Surface((self.player_radius * 4, self.player_radius * 4), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (self.player_radius*2, self.player_radius*2), self.player_radius * 2)
            surface.blit(glow_surf, (pos_int[0] - self.player_radius*2, pos_int[1] - self.player_radius*2), special_flags=pygame.BLEND_RGBA_ADD)
            # Player circle
            pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], self.player_radius, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], self.player_radius, self.COLOR_PLAYER)

    def _render_monsters(self, surface):
        for m in self.monsters:
            pos_int = (int(m['pos'][0]), int(m['pos'][1]))
            # Pulsating size
            pulse = math.sin(self.steps * 0.2 + m['pos'][0]) * 0.5 + 0.5 # Add pos to desync
            radius = int(m['radius'] * (1 + pulse * 0.1))
            pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], radius, m['color'])
            pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], radius, m['color'])
            # Health bar
            if m['health'] < m['max_health']:
                bar_w = m['radius'] * 2
                bar_h = 4
                bar_x = pos_int[0] - m['radius']
                bar_y = pos_int[1] - m['radius'] - 8
                health_ratio = m['health'] / m['max_health']
                pygame.draw.rect(surface, self.HEALTH_BAR_BG, (bar_x, bar_y, bar_w, bar_h))
                pygame.draw.rect(surface, self.HEALTH_BAR_FG, (bar_x, bar_y, bar_w * health_ratio, bar_h))

    def _render_projectiles(self, surface):
        for p in self.projectiles:
            pos_int = (int(p['pos'][0]), int(p['pos'][1]))
            end_pos_int = (int(p['pos'][0] - p['vel'][0]*0.8), int(p['pos'][1] - p['vel'][1]*0.8))
            pygame.draw.line(surface, self.COLOR_PROJECTILE, pos_int, end_pos_int, 3)

    def _render_particles(self, surface):
        for p in self.particles:
            if p['radius'] > 0 and p['lifespan'] > 0:
                pos_int = (int(p['pos'][0]), int(p['pos'][1]))
                
                max_lifespan = p.get('max_lifespan', 1)
                if max_lifespan <= 0: max_lifespan = 1

                fade_ratio = max(0.0, p['lifespan'] / max_lifespan)
                
                # Manually blend particle color with background to simulate fading.
                # This creates a valid (r, g, b) tuple for gfxdraw, fixing the error.
                r = int(p['color'][0] * fade_ratio + self.COLOR_BG[0] * (1 - fade_ratio))
                g = int(p['color'][1] * fade_ratio + self.COLOR_BG[1] * (1 - fade_ratio))
                b = int(p['color'][2] * fade_ratio + self.COLOR_BG[2] * (1 - fade_ratio))
                final_color = (r, g, b)

                pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], int(p['radius']), final_color)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        # Wave
        wave_text = self.font_small.render(f"WAVE: {self.wave}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))
        # Health
        health_text = self.font_small.render("HEALTH:", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 35))
        for i in range(self.player_max_health):
            color = self.COLOR_PLAYER if i < self.player_health else self.HEALTH_BAR_BG
            pygame.draw.rect(self.screen, color, (health_text.get_width() + 20 + i * 25, 35, 20, 15))
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            game_over_text = self.font_large.render("GAME OVER", True, self.COLOR_PLAYER)
            text_rect = game_over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(game_over_text, text_rect)
            
            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            score_rect = final_score_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
            self.screen.blit(final_score_text, score_rect)

if __name__ == '__main__':
    # This block allows you to play the game directly
    # To run, you will need to unset the dummy video driver
    # comment out or remove the line: os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    # or alternatively:
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Arcade Arena")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Re-initialize prev_space_held for human play
    env.prev_space_held = False

    while running:
        # --- Human Controls ---
        movement = 0 # no-op
        space_held = False
        shift_held = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = True
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = True
        if keys[pygame.K_r]: # Reset key
             obs, info = env.reset()
             total_reward = 0
             env.prev_space_held = False
             continue

        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # To auto-quit on game over, uncomment the following line
            # running = False
        
        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    pygame.quit()