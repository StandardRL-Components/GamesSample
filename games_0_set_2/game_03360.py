
# Generated: 2025-08-27T23:08:11.838629
# Source Brief: brief_03360.md
# Brief Index: 3360

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ←→ to move. Press ↑ or ↓ to dodge. Press space to attack."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defeat waves of monsters in a side-view fighter. Dodge attacks and strike at the right moment to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GROUND_HEIGHT = 50

    # Colors
    COLOR_BG = (25, 20, 40)
    COLOR_GROUND = (45, 40, 60)
    COLOR_PLAYER = (50, 220, 100)
    COLOR_PLAYER_DODGE = (150, 255, 200)
    COLOR_MONSTER = (230, 60, 60)
    COLOR_MONSTER_HURT = (255, 200, 200)
    COLOR_TELEGRAPH = (255, 220, 0)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_UI_BG = (10, 10, 20, 180)
    COLOR_HEALTH_GREEN = (40, 180, 80)
    COLOR_HEALTH_RED = (180, 40, 40)
    COLOR_HEALTH_BG = (60, 60, 60)
    
    # Game parameters
    MAX_STEPS = 1500
    MONSTERS_TO_WIN = 5
    
    # Player
    PLAYER_HEALTH_MAX = 100
    PLAYER_SPEED = 8
    PLAYER_DODGE_DURATION = 12 # frames
    PLAYER_DODGE_SPEED = 12
    PLAYER_DODGE_COOLDOWN = 40
    PLAYER_ATTACK_DURATION = 15
    PLAYER_ATTACK_COOLDOWN = 30
    PLAYER_ATTACK_RANGE = 70
    PLAYER_ATTACK_DAMAGE = 10
    PLAYER_HURT_DURATION = 15
    
    # Monster
    MONSTER_HEALTH_MAX = 50
    MONSTER_DAMAGE = 20
    MONSTER_HURT_DURATION = 8

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.font_float = pygame.font.Font(None, 28)

        self.player = {}
        self.monster = None
        self.particles = []
        self.floating_texts = []
        self.screen_shake = 0

        self.reset()

        # self.validate_implementation() # Optional validation call

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.reward_this_step = 0
        self.game_over = False
        self.monsters_defeated = 0

        self.player = {
            "pos": np.array([150.0, self.SCREEN_HEIGHT - self.GROUND_HEIGHT]),
            "size": np.array([30, 50]),
            "health": self.PLAYER_HEALTH_MAX,
            "state": "idle", # idle, dodging, attacking, hurt
            "state_timer": 0,
            "dodge_cooldown": 0,
            "attack_cooldown": 0,
        }
        self.last_space_held = False
        
        self.monster = None
        self._spawn_monster()
        
        self.particles = []
        self.floating_texts = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)
            
        self.reward_this_step = 0
        terminated = False
        
        if not self.game_over:
            self._handle_input(action)
            self._update_player()
            self._update_monster()
            self._handle_collisions()
            self._update_effects()

        self.steps += 1
        
        # Check termination conditions
        if self.player["health"] <= 0 and not self.game_over:
            self.reward_this_step -= 100
            self.game_over = True
            self._create_floating_text("DEFEAT", self.screen.get_rect().center, (200, 50, 50), 60)
        
        if self.monsters_defeated >= self.MONSTERS_TO_WIN and not self.game_over:
            self.reward_this_step += 100
            self.game_over = True
            self._create_floating_text("VICTORY!", self.screen.get_rect().center, (50, 200, 50), 60)

        if self.steps >= self.MAX_STEPS:
            terminated = True

        terminated = terminated or self.game_over
        self.score += self.reward_this_step

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        self.last_space_held = space_held

        is_action_taken = False

        # State-independent cooldowns
        if self.player["dodge_cooldown"] > 0: self.player["dodge_cooldown"] -= 1
        if self.player["attack_cooldown"] > 0: self.player["attack_cooldown"] -= 1

        if self.player["state"] == "idle":
            # Dodge
            if (movement == 1 or movement == 2) and self.player["dodge_cooldown"] == 0:
                self.player["state"] = "dodging"
                self.player["state_timer"] = self.PLAYER_DODGE_DURATION
                self.player["dodge_cooldown"] = self.PLAYER_DODGE_COOLDOWN
                is_action_taken = True
            # Attack
            elif space_pressed and self.player["attack_cooldown"] == 0:
                self.player["state"] = "attacking"
                self.player["state_timer"] = self.PLAYER_ATTACK_DURATION
                self.player["attack_cooldown"] = self.PLAYER_ATTACK_COOLDOWN
                # sfx: player_swing.wav
                is_action_taken = True
            # Movement
            elif movement == 3: # Left
                self.player["pos"][0] -= self.PLAYER_SPEED
                is_action_taken = True
            elif movement == 4: # Right
                self.player["pos"][0] += self.PLAYER_SPEED
                is_action_taken = True
        
        if not is_action_taken:
            self.reward_this_step -= 0.02

    def _update_player(self):
        # Update state timer
        if self.player["state_timer"] > 0:
            self.player["state_timer"] -= 1
            if self.player["state_timer"] == 0:
                self.player["state"] = "idle"

        # Dodge movement
        if self.player["state"] == "dodging":
            direction = 1 if self.player["pos"][0] < self.SCREEN_WIDTH / 2 else -1
            self.player["pos"][0] += direction * self.PLAYER_DODGE_SPEED

        # Clamp player position
        self.player["pos"][0] = np.clip(self.player["pos"][0], 0, self.SCREEN_WIDTH - self.player["size"][0])
        
    def _spawn_monster(self):
        difficulty_mod = 1.0 + self.monsters_defeated * 0.1
        attack_cooldown = max(30, 120 / difficulty_mod)

        self.monster = {
            "pos": np.array([self.SCREEN_WIDTH - 150.0, self.SCREEN_HEIGHT - self.GROUND_HEIGHT]),
            "size": np.array([40, 60]),
            "health": self.MONSTER_HEALTH_MAX,
            "state": "idle", # idle, telegraph, attack, hurt
            "state_timer": 0,
            "attack_type": self.np_random.choice(["lunge", "projectile"]),
            "attack_cooldown": self.np_random.integers(60, 90),
            "attack_cooldown_max": attack_cooldown,
            "projectiles": [],
            "telegraph_target": None,
        }

    def _update_monster(self):
        if not self.monster:
            return

        # Update state timer
        if self.monster["state_timer"] > 0:
            self.monster["state_timer"] -= 1
            if self.monster["state_timer"] == 0:
                # Execute attack after telegraph
                if self.monster["state"] == "telegraph":
                    self.monster["state"] = "attack"
                    if self.monster["attack_type"] == "lunge":
                        self.monster["state_timer"] = 20 # Lunge duration
                        # sfx: monster_lunge.wav
                    elif self.monster["attack_type"] == "projectile":
                        self.monster["state_timer"] = 10 # Fire pose
                        self._fire_projectile()
                else:
                    self.monster["state"] = "idle"

        if self.monster["state"] == "idle":
            self.monster["attack_cooldown"] -= 1
            if self.monster["attack_cooldown"] <= 0:
                self.monster["state"] = "telegraph"
                self.monster["state_timer"] = 30 # Telegraph duration
                # sfx: monster_charge.wav
                if self.monster["attack_type"] == "lunge":
                    self.monster["telegraph_target"] = self.player["pos"][0]
                elif self.monster["attack_type"] == "projectile":
                    self.monster["telegraph_target"] = self.player["pos"].copy()

        # Lunge movement
        if self.monster["state"] == "attack" and self.monster["attack_type"] == "lunge":
            target_x = self.monster["telegraph_target"]
            current_x = self.monster["pos"][0]
            direction = np.sign(target_x - current_x)
            self.monster["pos"][0] += direction * 15 # Lunge speed
        
        # Reset attack cycle
        if self.monster["state"] == "idle" and self.monster["attack_cooldown"] <= 0:
            self.monster["attack_cooldown"] = self.monster["attack_cooldown_max"]
            self.monster["attack_type"] = self.np_random.choice(["lunge", "projectile"])

        # Update projectiles
        for proj in self.monster["projectiles"]:
            proj["pos"] += proj["vel"]
        self.monster["projectiles"] = [p for p in self.monster["projectiles"] if 0 < p["pos"][0] < self.SCREEN_WIDTH]

    def _fire_projectile(self):
        # sfx: monster_shoot.wav
        start_pos = self.monster["pos"] + np.array([-10, -self.monster["size"][1] / 2])
        direction = self.player["pos"] - start_pos
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        
        projectile = {
            "pos": start_pos,
            "vel": direction * 10,
            "size": 12,
        }
        self.monster["projectiles"].append(projectile)

    def _handle_collisions(self):
        if not self.monster: return

        player_rect = pygame.Rect(self.player["pos"], self.player["size"])
        monster_rect = pygame.Rect(self.monster["pos"], self.monster["size"])

        # Player attacks monster
        if self.player["state"] == "attacking" and self.player["state_timer"] > (self.PLAYER_ATTACK_DURATION - 5): # Active frames
            attack_center_x = self.player["pos"][0] + self.player["size"][0] / 2
            attack_rect = pygame.Rect(attack_center_x, self.player["pos"][1] - self.player["size"][1], self.PLAYER_ATTACK_RANGE, self.player["size"][1])
            if attack_rect.colliderect(monster_rect):
                if self.monster["state"] != "hurt":
                    self.monster["health"] -= self.PLAYER_ATTACK_DAMAGE
                    self.reward_this_step += 1
                    self.monster["state"] = "hurt"
                    self.monster["state_timer"] = self.MONSTER_HURT_DURATION
                    self._create_particles(monster_rect.center, 15, self.COLOR_MONSTER)
                    self._create_floating_text(f"-{self.PLAYER_ATTACK_DAMAGE}", monster_rect.midtop, (255, 200, 100))
                    # sfx: hit_flesh.wav
                    if self.monster["health"] <= 0:
                        self.reward_this_step += 10
                        self.score += 100 # Separate from reward
                        self.monsters_defeated += 1
                        self._create_particles(monster_rect.center, 50, self.COLOR_MONSTER)
                        self.monster = None
                        if self.monsters_defeated < self.MONSTERS_TO_WIN:
                            self._spawn_monster()
                        return # Monster is gone, no more collision checks

        # Monster attacks player
        is_dodging = self.player["state"] == "dodging"
        
        # Lunge attack
        if self.monster["state"] == "attack" and self.monster["attack_type"] == "lunge":
            if monster_rect.colliderect(player_rect):
                if is_dodging:
                    if self.player["state_timer"] > 0: # Check if dodge is still active
                        self.reward_this_step += 0.1
                        self._create_floating_text("DODGE!", self.player["pos"] - [0, 20], (100, 200, 255))
                        self.player["state_timer"] = 0 # End dodge on successful use
                elif self.player["state"] != "hurt":
                    self._damage_player(self.MONSTER_DAMAGE, monster_rect.center)
        
        # Projectile attack
        for proj in self.monster["projectiles"]:
            proj_rect = pygame.Rect(proj["pos"][0] - proj["size"]/2, proj["pos"][1] - proj["size"]/2, proj["size"], proj["size"])
            if proj_rect.colliderect(player_rect):
                if is_dodging:
                    self.reward_this_step += 0.1
                    self._create_floating_text("DODGE!", self.player["pos"] - [0, 20], (100, 200, 255))
                elif self.player["state"] != "hurt":
                    self._damage_player(self.MONSTER_DAMAGE, proj["pos"])
                
                # Remove projectile on hit/dodge
                self.monster["projectiles"].remove(proj)
                self._create_particles(proj["pos"], 10, self.COLOR_TELEGRAPH)
                break

    def _damage_player(self, amount, origin_pos):
        self.player["health"] -= amount
        self.reward_this_step -= 1
        self.player["state"] = "hurt"
        self.player["state_timer"] = self.PLAYER_HURT_DURATION
        self.screen_shake = 10
        self._create_particles(self.player["pos"] + [self.player["size"][0]/2, -self.player["size"][1]/2], 20, self.COLOR_PLAYER)
        self._create_floating_text(f"-{amount}", self.player["pos"] - [0, 20], (255, 100, 100))
        # sfx: player_hurt.wav

    def _update_effects(self):
        # Particles
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]
        
        # Floating texts
        for t in self.floating_texts:
            t["pos"][1] -= 1
            t["life"] -= 1
        self.floating_texts = [t for t in self.floating_texts if t["life"] > 0]
        
        # Screen shake
        if self.screen_shake > 0:
            self.screen_shake -= 1

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": self.np_random.integers(15, 30),
                "color": color,
                "size": self.np_random.integers(2, 5)
            })

    def _create_floating_text(self, text, pos, color, lifetime=45):
        self.floating_texts.append({
            "text": text,
            "pos": list(pos),
            "color": color,
            "life": lifetime,
            "max_life": lifetime
        })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # Screen shake offset
        offset_x, offset_y = 0, 0
        if self.screen_shake > 0:
            offset_x = self.np_random.integers(-5, 6)
            offset_y = self.np_random.integers(-5, 6)
        
        # Render ground
        ground_rect = pygame.Rect(0, self.SCREEN_HEIGHT - self.GROUND_HEIGHT, self.SCREEN_WIDTH, self.GROUND_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GROUND, ground_rect.move(offset_x, offset_y))
        
        self._render_monster(offset_x, offset_y)
        self._render_player(offset_x, offset_y)
        self._render_effects(offset_x, offset_y)
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_player(self, ox, oy):
        p = self.player
        color = self.COLOR_PLAYER
        if p["state"] == "dodging": color = self.COLOR_PLAYER_DODGE
        if p["state"] == "hurt" and self.steps % 4 < 2: color = (255, 255, 255)

        player_rect = pygame.Rect(p["pos"], p["size"])
        player_rect.bottom = self.SCREEN_HEIGHT - self.GROUND_HEIGHT
        
        pygame.draw.rect(self.screen, color, player_rect.move(ox, oy))

        # Attack animation
        if p["state"] == "attacking":
            progress = 1 - (p["state_timer"] / self.PLAYER_ATTACK_DURATION)
            arc_rect = pygame.Rect(0, 0, self.PLAYER_ATTACK_RANGE * 2, p["size"][1] * 2)
            arc_rect.center = (player_rect.centerx, player_rect.centery - 10)
            
            start_angle = math.pi * (1 - progress * 2)
            end_angle = start_angle + math.pi/2
            if progress > 0.5:
                start_angle = math.pi * (1 - (1 - progress) * 2)
                end_angle = start_angle + math.pi/2

            pygame.draw.arc(self.screen, (200, 220, 255), arc_rect.move(ox, oy), start_angle, end_angle, 4)

    def _render_monster(self, ox, oy):
        if not self.monster: return
        m = self.monster
        color = self.COLOR_MONSTER
        if m["state"] == "hurt" and self.steps % 4 < 2: color = self.COLOR_MONSTER_HURT
        
        monster_rect = pygame.Rect(m["pos"], m["size"])
        monster_rect.bottom = self.SCREEN_HEIGHT - self.GROUND_HEIGHT
        pygame.draw.rect(self.screen, color, monster_rect.move(ox, oy))

        # Telegraph
        if m["state"] == "telegraph":
            alpha = int(255 * (math.sin(self.steps * 0.5)**2))
            color_telegraph = self.COLOR_TELEGRAPH + (alpha,)
            
            if m["attack_type"] == "lunge":
                start_pos = (monster_rect.centerx + ox, monster_rect.top + oy)
                end_pos = (m["telegraph_target"] + self.player["size"][0]/2 + ox, monster_rect.top + oy)
                pygame.draw.line(self.screen, self.COLOR_TELEGRAPH, start_pos, end_pos, 3)
            elif m["attack_type"] == "projectile":
                pygame.gfxdraw.filled_circle(self.screen, int(m["pos"][0] - 10 + ox), int(m["pos"][1] - m["size"][1]/2 + oy), 8, color_telegraph)

        # Projectiles
        for proj in m["projectiles"]:
            pygame.draw.circle(self.screen, self.COLOR_TELEGRAPH, (int(proj["pos"][0] + ox), int(proj["pos"][1] + oy)), proj["size"] // 2)

    def _render_effects(self, ox, oy):
        # Particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["life"]))
            color = p["color"] + (alpha,)
            pos = (int(p["pos"][0] + ox), int(p["pos"][1] + oy))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], p["size"], color)

        # Floating texts
        for t in self.floating_texts:
            alpha = int(255 * (t["life"] / t["max_life"]))
            text_surf = self.font_float.render(t["text"], True, t["color"])
            text_surf.set_alpha(alpha)
            self.screen.blit(text_surf, (t["pos"][0] - text_surf.get_width()/2, t["pos"][1]))
            
    def _render_ui(self):
        # Player Health
        hp_bar_width = 200
        hp_bar_rect_bg = pygame.Rect(10, 10, hp_bar_width, 25)
        hp_ratio = max(0, self.player["health"] / self.PLAYER_HEALTH_MAX)
        hp_bar_rect_fill = pygame.Rect(10, 10, int(hp_bar_width * hp_ratio), 25)
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, hp_bar_rect_bg)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, hp_bar_rect_fill)
        
        # Monster Health
        if self.monster:
            mhp_bar_width = 200
            mhp_bar_rect_bg = pygame.Rect(self.SCREEN_WIDTH - mhp_bar_width - 10, 10, mhp_bar_width, 25)
            mhp_ratio = max(0, self.monster["health"] / self.MONSTER_HEALTH_MAX)
            mhp_bar_rect_fill = pygame.Rect(self.SCREEN_WIDTH - int(mhp_bar_width * mhp_ratio) - 10, 10, int(mhp_bar_width * mhp_ratio), 25)
            
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, mhp_bar_rect_bg)
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, mhp_bar_rect_fill)

        # Monster Count
        count_text = f"Monsters: {self.monsters_defeated} / {self.MONSTERS_TO_WIN}"
        text_surf = self.font_main.render(count_text, True, self.COLOR_UI_TEXT)
        text_rect = text_surf.get_rect(centerx=self.SCREEN_WIDTH/2, y=10)
        self.screen.blit(text_surf, text_rect)
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player["health"],
            "monsters_defeated": self.monsters_defeated,
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game with keyboard controls
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Mapping from Pygame keys to MultiDiscrete actions
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Pygame setup for human play
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Monster Fighter")
    clock = pygame.time.Clock()

    total_reward = 0
    
    while not done:
        # --- Human Input ---
        movement_action = 0 # No-op
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        for key, move_val in key_map.items():
            if keys[key]:
                movement_action = move_val
                break # Prioritize first key in map
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1
            
        action = [movement_action, space_action, shift_action]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Render to screen ---
        # The observation is (H, W, C), but pygame needs (W, H) surface
        # So we transpose it back
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            done = True
        
        clock.tick(30)
        
    env.close()