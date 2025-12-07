
# Generated: 2025-08-28T03:46:23.005038
# Source Brief: brief_02121.md
# Brief Index: 2121

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: Use ←→ to move. Press ↑ to jump. Press space to attack monsters."
    )

    game_description = (
        "Jump and slash your way through hordes of monsters in this side-scrolling arcade action game. "
        "Defeat 15 monsters to win, but watch your health!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.game_font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # Game constants
        self.gravity = 0.8
        self.ground_y = self.screen_height - 60
        self.max_steps = 2000
        self.win_condition_kills = 15

        # Colors
        self.color_bg_top = (10, 5, 25)
        self.color_bg_bottom = (40, 20, 60)
        self.color_ground = (30, 15, 50)
        self.color_player = (50, 255, 50)
        self.color_player_hit = (255, 100, 100)
        self.color_slash = (255, 255, 100)
        self.color_heart = (255, 0, 0)
        self.color_text = (255, 255, 255)
        self.monster_colors = {
            'walker': (255, 50, 50),
            'jumper': (200, 50, 255),
            'shooter': (50, 150, 255)
        }
        self.projectile_color = (255, 100, 200)

        # State variables will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.monsters_killed = 0
        self.game_over = False
        self.player = {}
        self.monsters = []
        self.projectiles = []
        self.particles = []
        self.screen_shake = 0
        self.prev_space_held = False

        self.reset()
        
        # This check is disabled by default to avoid printing on import
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.monsters_killed = 0
        self.game_over = False

        self.player = {
            "pos": pygame.Vector2(self.screen_width / 2, self.ground_y),
            "vel": pygame.Vector2(0, 0),
            "size": pygame.Vector2(30, 40),
            "on_ground": True,
            "health": 3,
            "invulnerable_timer": 0,
            "attack_timer": 0,
            "attack_cooldown": 0,
            "facing_right": True,
            "squash": 0,
        }
        
        self.monsters.clear()
        self.projectiles.clear()
        self.particles.clear()
        
        self.spawn_timer = 60 # Start spawning after 2 seconds
        self.spawn_interval = 60 # 2 seconds at 30fps

        self.prev_space_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        # --- Handle Input ---
        self._handle_input(action)

        # --- Update Game State ---
        reward += self._update_player()
        self._update_monsters()
        self._update_projectiles()
        reward += self._handle_collisions()
        self._update_particles()

        # --- Update Timers ---
        if self.spawn_timer > 0:
            self.spawn_timer -= 1
        else:
            self._spawn_monster()
            # Difficulty scaling
            spawn_reduction = (self.monsters_killed // 5) * 3
            self.spawn_interval = max(30, 60 - spawn_reduction) # Min 1s spawn time
            self.spawn_timer = self.spawn_interval

        if self.screen_shake > 0:
            self.screen_shake -= 1

        # --- Termination Check ---
        terminated = False
        if self.player["health"] <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
        elif self.monsters_killed >= self.win_condition_kills:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.max_steps:
            terminated = True
            self.game_over = True
            
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Horizontal Movement
        if movement == 3:  # Left
            self.player["vel"].x = -5
            self.player["facing_right"] = False
        elif movement == 4:  # Right
            self.player["vel"].x = 5
            self.player["facing_right"] = True
        else:
            self.player["vel"].x = 0

        # Jump
        if movement == 1 and self.player["on_ground"]:  # Up
            self.player["vel"].y = -15
            self.player["on_ground"] = False
            self.player["squash"] = 10 # For jump animation
            # sfx: player_jump

        # Attack
        if space_held and not self.prev_space_held and self.player["attack_cooldown"] <= 0:
            self.player["attack_timer"] = 8  # Attack lasts for 8 frames
            self.player["attack_cooldown"] = 15 # Cooldown of 0.5s
            # sfx: player_attack_swing
        
        self.prev_space_held = space_held

    def _update_player(self):
        # Apply gravity
        self.player["vel"].y += self.gravity
        
        # Move player
        self.player["pos"] += self.player["vel"]
        
        # Ground collision
        landed = False
        if self.player["pos"].y > self.ground_y:
            if not self.player["on_ground"]:
                landed = True
            self.player["pos"].y = self.ground_y
            self.player["vel"].y = 0
            self.player["on_ground"] = True
        
        if landed:
            self.player["squash"] = 10 # For landing animation
            # sfx: player_land
        
        # Screen boundaries
        self.player["pos"].x = max(0, min(self.screen_width - self.player["size"].x, self.player["pos"].x))

        # Update timers
        if self.player["invulnerable_timer"] > 0: self.player["invulnerable_timer"] -= 1
        if self.player["attack_timer"] > 0: self.player["attack_timer"] -= 1
        if self.player["attack_cooldown"] > 0: self.player["attack_cooldown"] -= 1
        if self.player["squash"] > 0: self.player["squash"] -= 1
        
        # Continuous reward for being on ground near monster
        reward = 0
        if self.player["on_ground"]:
            for m in self.monsters:
                dist = abs(self.player["pos"].x - m["pos"].x)
                if dist < 100:
                    reward -= 0.02 # Small continuous penalty, capped per step by monster count
        return reward

    def _update_monsters(self):
        for m in self.monsters:
            if m['type'] == 'walker':
                m['pos'].x += m['vel'].x
                if m['pos'].x <= 0 or m['pos'].x >= self.screen_width - m['size'].x:
                    m['vel'].x *= -1
            elif m['type'] == 'jumper':
                if m['on_ground']:
                    m['cooldown'] -= 1
                    if m['cooldown'] <= 0:
                        m['vel'].y = -12
                        m['on_ground'] = False
                        m['cooldown'] = self.np_random.integers(60, 120)
                else:
                    m['vel'].y += self.gravity
                    m['pos'] += m['vel']
                    if m['pos'].y >= self.ground_y:
                        m['pos'].y = self.ground_y
                        m['vel'].y = 0
                        m['on_ground'] = True
            elif m['type'] == 'shooter':
                m['cooldown'] -= 1
                if m['cooldown'] <= 0:
                    # sfx: monster_shoot
                    direction = -1 if self.player['pos'].x < m['pos'].x else 1
                    proj_vel = pygame.Vector2(6 * direction, 0)
                    self.projectiles.append({
                        "pos": m['pos'].copy() + pygame.Vector2(m['size'].x/2 * direction, m['size'].y/2),
                        "vel": proj_vel,
                        "size": 8
                    })
                    m['cooldown'] = self.np_random.integers(90, 150)

    def _spawn_monster(self):
        if len(self.monsters) >= 8: return

        monster_type = self.np_random.choice(['walker', 'jumper', 'shooter'])
        side = self.np_random.choice([-1, 1])
        x_pos = -30 if side == -1 else self.screen_width + 30
        
        monster = {
            "pos": pygame.Vector2(x_pos, self.ground_y),
            "vel": pygame.Vector2(0, 0),
            "health": 1,
        }

        if monster_type == 'walker':
            monster.update({
                "type": 'walker', "size": pygame.Vector2(40, 30),
                "vel": pygame.Vector2(2 * side * -1, 0)
            })
        elif monster_type == 'jumper':
            monster.update({
                "type": 'jumper', "size": pygame.Vector2(30, 30),
                "on_ground": True, "cooldown": self.np_random.integers(30, 90)
            })
        elif monster_type == 'shooter':
             monster.update({
                "type": 'shooter', "size": pygame.Vector2(35, 35),
                "pos": pygame.Vector2(x_pos, self.ground_y - 5),
                "cooldown": self.np_random.integers(60, 120)
            })
        
        self.monsters.append(monster)

    def _update_projectiles(self):
        self.projectiles[:] = [p for p in self.projectiles if 0 < p['pos'].x < self.screen_width]
        for p in self.projectiles:
            p['pos'] += p['vel']

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player["pos"], self.player["size"])

        # Player attack vs Monsters
        if self.player["attack_timer"] > 0:
            attack_range = 50
            attack_center_y = self.player["pos"].y + self.player["size"].y / 2
            direction = 1 if self.player["facing_right"] else -1
            attack_rect = pygame.Rect(
                self.player["pos"].x + (self.player["size"].x if direction == 1 else -attack_range),
                attack_center_y - attack_range / 2,
                attack_range,
                attack_range
            )
            
            for m in self.monsters[:]:
                monster_rect = pygame.Rect(m["pos"], m["size"])
                if attack_rect.colliderect(monster_rect):
                    self.monsters.remove(m)
                    reward += 1.0
                    self.score += 1
                    self.monsters_killed += 1
                    # sfx: monster_hit
                    self._create_particles(m['pos'] + m['size']/2, self.monster_colors[m['type']], 20)

        # Monsters/Projectiles vs Player
        if self.player["invulnerable_timer"] <= 0:
            for m in self.monsters:
                monster_rect = pygame.Rect(m["pos"], m["size"])
                if player_rect.colliderect(monster_rect):
                    self._damage_player(1)
                    reward -= 1.0
                    break
            
            if self.player["invulnerable_timer"] <= 0:
                for p in self.projectiles[:]:
                    proj_rect = pygame.Rect(p['pos'].x - p['size']/2, p['pos'].y - p['size']/2, p['size'], p['size'])
                    if player_rect.colliderect(proj_rect):
                        self._damage_player(1)
                        reward -= 1.0
                        self.projectiles.remove(p)
                        break
        return reward
    
    def _damage_player(self, amount):
        self.player["health"] -= amount
        self.player["invulnerable_timer"] = 60 # 2 seconds invulnerability
        self.screen_shake = 10
        # sfx: player_hit
        self._create_particles(self.player["pos"] + self.player["size"]/2, self.color_player_hit, 15)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append({
                "pos": pos.copy(),
                "vel": pygame.Vector2(self.np_random.uniform(-4, 4), self.np_random.uniform(-4, 4)),
                "life": self.np_random.integers(10, 20),
                "color": color,
                "size": self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        self.particles[:] = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['life'] -= 1

    def _get_observation(self):
        # --- Background ---
        for i in range(self.screen_height):
            ratio = i / self.screen_height
            color = (
                self.color_bg_top[0] * (1 - ratio) + self.color_bg_bottom[0] * ratio,
                self.color_bg_top[1] * (1 - ratio) + self.color_bg_bottom[1] * ratio,
                self.color_bg_top[2] * (1 - ratio) + self.color_bg_bottom[2] * ratio,
            )
            pygame.draw.line(self.screen, color, (0, i), (self.screen_width, i))
        
        # --- Ground ---
        pygame.draw.rect(self.screen, self.color_ground, (0, self.ground_y, self.screen_width, self.screen_height - self.ground_y))

        # --- Game Elements ---
        self._render_particles()
        self._render_monsters()
        self._render_projectiles()
        self._render_player()
        
        # --- UI ---
        self._render_ui()

        # --- Screen Shake ---
        obs_surface = self.screen
        if self.screen_shake > 0:
            shake_surface = pygame.Surface((self.screen_width, self.screen_height))
            offset_x = self.np_random.integers(-5, 5)
            offset_y = self.np_random.integers(-5, 5)
            shake_surface.blit(self.screen, (offset_x, offset_y))
            obs_surface = shake_surface

        arr = pygame.surfarray.array3d(obs_surface)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_player(self):
        # Invulnerability blink
        if self.player["invulnerable_timer"] > 0 and (self.player["invulnerable_timer"] // 3) % 2 == 0:
            return

        # Squash and stretch animation
        squash_factor = self.player["squash"] / 20.0
        width = self.player["size"].x * (1 + squash_factor)
        height = self.player["size"].y * (1 - squash_factor)
        pos_x = self.player["pos"].x - (width - self.player["size"].x) / 2
        pos_y = self.player["pos"].y + (self.player["size"].y - height)

        player_rect = pygame.Rect(pos_x, pos_y, width, height)
        pygame.draw.rect(self.screen, self.color_player, player_rect, border_radius=4)
        
        # Attack visual
        if self.player["attack_timer"] > 0:
            progress = 1.0 - (self.player["attack_timer"] / 8.0)
            direction = 1 if self.player["facing_right"] else -1
            
            start_angle = -90 if direction == 1 else 90
            end_angle = 90 if direction == 1 else 270
            
            current_angle = start_angle + (end_angle - start_angle) * progress
            
            center_x = int(self.player["pos"].x + self.player["size"].x / 2)
            center_y = int(self.player["pos"].y + self.player["size"].y / 2)
            radius = 35
            
            arc_points = []
            num_points = 10
            for i in range(num_points + 1):
                angle = math.radians(start_angle + (current_angle - start_angle) * (i / num_points))
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                arc_points.append((int(x), int(y)))
            
            if len(arc_points) > 1:
                pygame.draw.lines(self.screen, self.color_slash, False, arc_points, 4)

    def _render_monsters(self):
        for m in self.monsters:
            rect = pygame.Rect(m["pos"], m["size"])
            color = self.monster_colors[m['type']]
            if m['type'] == 'walker':
                pygame.draw.rect(self.screen, color, rect, border_radius=5)
            elif m['type'] == 'jumper':
                pygame.draw.ellipse(self.screen, color, rect)
            elif m['type'] == 'shooter':
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.color_bg_bottom, rect.inflate(-10, -10))

    def _render_projectiles(self):
        for p in self.projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['size']), self.projectile_color)
            pygame.gfxdraw.aacircle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['size']), self.projectile_color)

    def _render_particles(self):
        for p in self.particles:
            alpha = p['life'] / 20.0
            color = (*p['color'], int(alpha * 255))
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, p['pos'] - pygame.Vector2(p['size'], p['size']))

    def _render_ui(self):
        # Health
        for i in range(self.player["health"]):
            heart_pos = (20 + i * 40, 20)
            pygame.gfxdraw.filled_polygon(self.screen, [
                (heart_pos[0] + 15, heart_pos[1] + 30), (heart_pos[0], heart_pos[1] + 10),
                (heart_pos[0] + 15, heart_pos[1]), (heart_pos[0] + 30, heart_pos[1] + 10)
            ], self.color_heart)
        
        # Score
        score_text = self.game_font.render(f"Score: {self.score}", True, self.color_text)
        self.screen.blit(score_text, (self.screen_width - score_text.get_width() - 20, 20))
        
        # Win/Loss Message
        if self.game_over:
            if self.monsters_killed >= self.win_condition_kills:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
            
            end_text = self.game_font.render(msg, True, self.color_text)
            text_rect = end_text.get_rect(center=(self.screen_width/2, self.screen_height/2))
            self.screen.blit(end_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player["health"],
            "monsters_killed": self.monsters_killed,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Monster Slayer")
    
    terminated = False
    running = True
    total_reward = 0
    
    # Mapping keyboard keys to actions
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while running:
        # --- Player Input ---
        movement_action = 0 # No-op
        space_action = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        elif keys[pygame.K_UP]: movement_action = 1
        
        if keys[pygame.K_SPACE]: space_action = 1
        if keys[pygame.K_ESCAPE]: running = False
        
        action = [movement_action, space_action, 0] # Shift is unused

        # --- Game Step ---
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Check for reset ---
        if terminated and keys[pygame.K_r]:
            obs, info = env.reset()
            terminated = False
            total_reward = 0
        
        env.clock.tick(30) # Run at 30 FPS

    env.close()