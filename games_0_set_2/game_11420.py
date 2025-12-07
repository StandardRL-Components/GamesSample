import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Blast through a giant, segmented centipede in this retro-style arcade shooter. "
        "Destroy segments to gain XP, level up, and survive the ever-faster onslaught."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to move your ship. "
        "Press space to fire your weapon in the direction you last moved."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 5000

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (25, 25, 40)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_GLOW = (0, 255, 128)
    COLOR_CENTIPEDE = (255, 50, 100)
    COLOR_CENTIPEDE_GLOW = (255, 50, 100)
    COLOR_PROJECTILE = (100, 200, 255)
    COLOR_PROJECTILE_GLOW = (100, 200, 255)
    COLOR_PARTICLE = (255, 200, 50)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_HP_BAR = (50, 205, 50)
    COLOR_HP_BAR_BG = (139, 0, 0)
    COLOR_XP_BAR = (64, 150, 255)
    COLOR_XP_BAR_BG = (40, 40, 80)

    # Player
    PLAYER_SPEED = 6
    PLAYER_RADIUS = 12
    PLAYER_MAX_HP = 100
    PLAYER_SHOOT_COOLDOWN = 5 # frames
    PLAYER_INVINCIBILITY_DURATION = 30 # frames

    # Projectile
    PROJECTILE_SPEED = 12
    PROJECTILE_RADIUS = 4

    # Centipede
    CENTIPEDE_SEGMENT_RADIUS = 10
    CENTIPEDE_INITIAL_SPEED = 1.5
    CENTIPEDE_SPEED_INCREASE_INTERVAL = 500
    CENTIPEDE_SPEED_INCREASE_AMOUNT = 0.05
    CENTIPEDE_TURN_STEP = 25

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        self.render_mode = render_mode
        # self._initialize_state() # This will be called in reset()

    def _initialize_state(self):
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Player state
        self.player_pos = [self.WIDTH / 2, self.HEIGHT - 50]
        self.player_hp = self.PLAYER_MAX_HP
        self.player_level = 1
        self.player_xp = 0
        self.player_xp_needed = 10
        self.player_shoot_cooldown_timer = 0
        self.player_invincibility_timer = 0
        self.last_move_direction = (0, -1) # Default up

        # Entity lists
        self.projectiles = []
        self.centipedes = []
        self.particles = []

        # Effects
        self.screen_shake_timer = 0

        # Difficulty
        self.centipede_speed = self.CENTIPEDE_INITIAL_SPEED

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state()

        # Spawn initial boss centipede
        self._spawn_centipede(
            pos=[self.WIDTH / 2, self.CENTIPEDE_TURN_STEP],
            length=15,
            is_boss=True
        )
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0.0

        # --- Handle Input & Player Update ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        move_vector = [0, 0]
        if movement == 1: move_vector[1] -= 1  # Up
        elif movement == 2: move_vector[1] += 1  # Down
        elif movement == 3: move_vector[0] -= 1  # Left
        elif movement == 4: move_vector[0] += 1  # Right
        
        if movement > 0:
            self.last_move_direction = tuple(move_vector)

        self.player_pos[0] += move_vector[0] * self.PLAYER_SPEED
        self.player_pos[1] += move_vector[1] * self.PLAYER_SPEED
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.HEIGHT - self.PLAYER_RADIUS)

        if self.player_shoot_cooldown_timer > 0: self.player_shoot_cooldown_timer -= 1
        if self.player_invincibility_timer > 0: self.player_invincibility_timer -= 1
        if self.screen_shake_timer > 0: self.screen_shake_timer -= 1

        if space_held and self.player_shoot_cooldown_timer == 0:
            # sfx: player_shoot
            self.projectiles.append({
                'pos': list(self.player_pos),
                'vel': self.last_move_direction,
                'radius': self.PROJECTILE_RADIUS
            })
            self.player_shoot_cooldown_timer = self.PLAYER_SHOOT_COOLDOWN

        # --- Update Game Entities ---
        self._update_projectiles()
        reward += self._update_centipedes()
        self._update_particles()
        reward += self._handle_collisions()
        self._update_difficulty()
        
        # --- Check Termination ---
        terminated = False
        truncated = False
        if self.player_hp <= 0:
            terminated = True
            self.game_over = True
        elif not any(c['is_boss'] for c in self.centipedes):
            reward += 100.0
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p['pos'][0] += p['vel'][0] * self.PROJECTILE_SPEED
            p['pos'][1] += p['vel'][1] * self.PROJECTILE_SPEED
            if not (0 < p['pos'][0] < self.WIDTH and 0 < p['pos'][1] < self.HEIGHT):
                self.projectiles.remove(p)

    def _update_centipedes(self):
        for centipede in self.centipedes:
            head = centipede['segments'][0]
            head['pos'][0] += centipede['dir'] * self.centipede_speed

            if not (self.CENTIPEDE_SEGMENT_RADIUS < head['pos'][0] < self.WIDTH - self.CENTIPEDE_SEGMENT_RADIUS):
                centipede['dir'] *= -1
                head['pos'][0] = np.clip(head['pos'][0], self.CENTIPEDE_SEGMENT_RADIUS, self.WIDTH - self.CENTIPEDE_SEGMENT_RADIUS)
                for seg in centipede['segments']:
                    seg['pos'][1] += self.CENTIPEDE_TURN_STEP
                    if seg['pos'][1] > self.HEIGHT - self.CENTIPEDE_SEGMENT_RADIUS:
                        seg['pos'][1] = self.HEIGHT - self.CENTIPEDE_SEGMENT_RADIUS

            for i in range(len(centipede['segments']) - 1, 0, -1):
                leader_pos = centipede['segments'][i-1]['pos']
                follower_pos = centipede['segments'][i]['pos']
                dist_x = leader_pos[0] - follower_pos[0]
                dist_y = leader_pos[1] - follower_pos[1]
                distance = math.hypot(dist_x, dist_y)
                
                target_dist = self.CENTIPEDE_SEGMENT_RADIUS * 1.8
                if distance > target_dist:
                    angle = math.atan2(dist_y, dist_x)
                    move_dist = distance - target_dist
                    follower_pos[0] += math.cos(angle) * move_dist
                    follower_pos[1] += math.sin(angle) * move_dist
        return 0

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0.0
        # Projectiles vs Centipedes
        for p in self.projectiles[:]:
            for centipede in self.centipedes[:]:
                for i, seg in enumerate(centipede['segments']):
                    dist = math.hypot(p['pos'][0] - seg['pos'][0], p['pos'][1] - seg['pos'][1])
                    if dist < self.PROJECTILE_RADIUS + self.CENTIPEDE_SEGMENT_RADIUS:
                        reward += 0.1
                        self.score += 10
                        self._spawn_particles(seg['pos'], self.COLOR_PARTICLE, 10, 2)
                        if p in self.projectiles: self.projectiles.remove(p)
                        
                        # sfx: segment_hit
                        reward += 1.0
                        self.score += 50
                        self._add_xp(1)
                        
                        remaining_segments = centipede['segments'][i+1:]
                        centipede['segments'] = centipede['segments'][:i]

                        if not centipede['segments']:
                            self.centipedes.remove(centipede)
                            reward += 10.0 # Boss segment destroyed
                            self.score += 200
                        
                        if remaining_segments:
                            self._spawn_centipede(
                                pos=remaining_segments[0]['pos'],
                                length=len(remaining_segments),
                                segments_data=remaining_segments,
                                direction=-centipede['dir'],
                                is_boss=centipede['is_boss']
                            )
                        break # Next projectile
                else: continue
                break # Next projectile
        
        # Player vs Centipedes
        if self.player_invincibility_timer == 0:
            for centipede in self.centipedes:
                for seg in centipede['segments']:
                    dist = math.hypot(self.player_pos[0] - seg['pos'][0], self.player_pos[1] - seg['pos'][1])
                    if dist < self.PLAYER_RADIUS + self.CENTIPEDE_SEGMENT_RADIUS:
                        # sfx: player_hit
                        self.player_hp -= 25
                        reward -= 0.1
                        self.player_invincibility_timer = self.PLAYER_INVINCIBILITY_DURATION
                        self.screen_shake_timer = 10
                        self._spawn_particles(self.player_pos, self.COLOR_PLAYER, 20, 4)
                        return reward # Exit after one hit
        return reward

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.CENTIPEDE_SPEED_INCREASE_INTERVAL == 0:
            self.centipede_speed += self.CENTIPEDE_SPEED_INCREASE_AMOUNT

    def _add_xp(self, amount):
        self.player_xp += amount
        if self.player_xp >= self.player_xp_needed:
            # sfx: level_up
            self.player_level += 1
            self.player_xp -= self.player_xp_needed
            self.player_xp_needed = int(self.player_xp_needed * 1.5)
            self.score += 100 * self.player_level
            self._spawn_particles(self.player_pos, self.COLOR_XP_BAR, 50, 5)

    def _get_observation(self):
        render_offset = [0, 0]
        if self.screen_shake_timer > 0:
            render_offset = [random.randint(-5, 5), random.randint(-5, 5)]

        # Background
        self.screen.fill(self.COLOR_BG)
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)

        # Game elements
        self._render_particles(render_offset)
        self._render_centipedes(render_offset)
        self._render_projectiles(render_offset)
        self._render_player(render_offset)

        # UI
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_player(self, offset):
        if self.player_invincibility_timer > 0 and self.steps % 4 < 2:
            return # Flashing effect
            
        pos = (int(self.player_pos[0] + offset[0]), int(self.player_pos[1] + offset[1]))
        
        # Glow
        glow_radius = int(self.PLAYER_RADIUS * 2.5)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.COLOR_PLAYER_GLOW, 30), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (pos[0] - glow_radius, pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _render_projectiles(self, offset):
        for p in self.projectiles:
            pos = (int(p['pos'][0] + offset[0]), int(p['pos'][1] + offset[1]))
            
            # Glow
            glow_radius = int(p['radius'] * 3)
            s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.COLOR_PROJECTILE_GLOW, 80), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(s, (pos[0] - glow_radius, pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], p['radius'], self.COLOR_PROJECTILE)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], p['radius'], self.COLOR_PROJECTILE)

    def _render_centipedes(self, offset):
        for centipede in self.centipedes:
            for seg in centipede['segments']:
                pos = (int(seg['pos'][0] + offset[0]), int(seg['pos'][1] + offset[1]))
                
                # Glow
                glow_radius = int(self.CENTIPEDE_SEGMENT_RADIUS * 1.8)
                s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*self.COLOR_CENTIPEDE_GLOW, 50), (glow_radius, glow_radius), glow_radius)
                self.screen.blit(s, (pos[0] - glow_radius, pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.CENTIPEDE_SEGMENT_RADIUS, self.COLOR_CENTIPEDE)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.CENTIPEDE_SEGMENT_RADIUS, self.COLOR_CENTIPEDE)

    def _render_particles(self, offset):
        for p in self.particles:
            pos = (int(p['pos'][0] + offset[0]), int(p['pos'][1] + offset[1]))
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = (*p['color'], alpha)
            pygame.draw.circle(self.screen, color, pos, int(p['size']))

    def _render_ui(self):
        # Score and Level
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        level_text = self.font_large.render(f"LV: {self.player_level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (10, 45))

        # HP Bar
        hp_bar_width = 200
        hp_ratio = max(0, self.player_hp / self.PLAYER_MAX_HP)
        pygame.draw.rect(self.screen, self.COLOR_HP_BAR_BG, (10, 10, hp_bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HP_BAR, (10, 10, hp_bar_width * hp_ratio, 20))

        # XP Bar
        xp_bar_width = 200
        xp_ratio = max(0, self.player_xp / self.player_xp_needed)
        pygame.draw.rect(self.screen, self.COLOR_XP_BAR_BG, (70, 50, xp_bar_width, 10))
        pygame.draw.rect(self.screen, self.COLOR_XP_BAR, (70, 50, xp_bar_width * xp_ratio, 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0,128))
            self.screen.blit(overlay, (0,0))
            
            win_condition = not any(c['is_boss'] for c in self.centipedes)
            end_text_str = "AREA CLEARED" if win_condition else "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, (255,255,255))
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.player_level}

    def _spawn_centipede(self, pos, length, segments_data=None, direction=1, is_boss=False):
        centipede = {
            'segments': [],
            'dir': direction,
            'is_boss': is_boss
        }
        if segments_data:
            centipede['segments'] = segments_data
        else:
            for i in range(length):
                centipede['segments'].append({
                    'pos': [pos[0] - i * self.CENTIPEDE_SEGMENT_RADIUS * 1.8 * direction, pos[1]],
                })
        self.centipedes.append(centipede)

    def _spawn_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, max_speed)
            life = random.randint(15, 30)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'color': color,
                'size': random.uniform(1, 4)
            })

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will create a window and let you control the agent
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Centipede RPG Gym Environment")
    
    terminated = False
    truncated = False
    total_reward = 0
    
    running = True
    while running and not (terminated or truncated):
        # --- Human Input ---
        movement = 0
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        # The observation is already the rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    
    # Keep window open for a bit to see the final state
    if terminated or truncated:
        start_time = pygame.time.get_ticks()
        while pygame.time.get_ticks() - start_time < 3000:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
            else:
                continue
            break

    env.close()