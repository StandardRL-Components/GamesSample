
# Generated: 2025-08-28T05:44:18.102186
# Source Brief: brief_02699.md
# Brief Index: 2699

        
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
        "Controls: ←→ to move. Hold Shift to fire your ranged weapon. Press Space to perform a melee attack."
    )

    game_description = (
        "Defeat waves of monsters in this side-scrolling action game. "
        "Use your limited ammo wisely and master the timing of your powerful melee strike."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and World Dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.WORLD_WIDTH = 1200
        
        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GROUND = (40, 35, 45)
        self.COLOR_PLAYER = (0, 200, 255)
        self.COLOR_PLAYER_GLOW = (0, 200, 255, 50)
        self.COLOR_AMMO = (0, 150, 255)
        self.COLOR_HEALTH_PLAYER = (0, 255, 100)
        self.COLOR_HEALTH_ENEMY = (255, 50, 50)
        self.COLOR_HEALTH_BG = (70, 70, 70)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_DAMAGE = (255, 255, 0)

        # Game constants
        self.MAX_STEPS = 1500 # Increased from 1000 to allow more time for later waves
        self.PLAYER_HEALTH_MAX = 100
        self.PLAYER_AMMO_MAX = 10
        self.PLAYER_SPEED = 4
        self.PLAYER_JUMP_STRENGTH = 10 # Unused in this design, but good for future
        self.GRAVITY = 0.5

        # Initialize state variables
        self.player_state = {}
        self.monsters = []
        self.player_projectiles = []
        self.monster_projectiles = []
        self.particles = []
        self.damage_popups = []
        self.melee_attack = None
        self.camera_x = 0
        self.screen_shake = 0
        self.last_action = np.array([0, 0, 0])
        self.steps = 0
        self.score = 0
        self.wave = 0
        self.game_over = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.wave = 0
        self.game_over = False
        
        self.player_state = {
            "pos": pygame.Vector2(self.WORLD_WIDTH / 2, self.SCREEN_HEIGHT - 50),
            "vel": pygame.Vector2(0, 0),
            "health": self.PLAYER_HEALTH_MAX,
            "ammo": self.PLAYER_AMMO_MAX,
            "facing_right": True,
            "on_ground": True,
            "melee_cooldown": 0,
            "ranged_cooldown": 0,
            "invuln_timer": 0,
        }

        self.monsters = []
        self.player_projectiles = []
        self.monster_projectiles = []
        self.particles = []
        self.damage_popups = []
        self.melee_attack = None
        self.last_action = np.array([0, 0, 0])
        self.screen_shake = 0

        self._spawn_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False

        if not self.game_over:
            # --- Action Handling ---
            space_pressed = space_held and not self.last_action[1]
            shift_pressed = shift_held and not self.last_action[2]

            # --- Update Game Logic ---
            reward += self._update_player(movement, space_pressed, shift_pressed)
            reward += self._update_monsters()
            self._update_projectiles()
            reward += self._handle_collisions()
            self._update_effects()

            # --- Check Win/Loss/Progression ---
            if not self.monsters:
                reward += 10  # Wave clear bonus
                self.score += 1000
                self._spawn_wave()
                self.player_state["health"] = min(self.PLAYER_HEALTH_MAX, self.player_state["health"] + 25) # Heal between waves
                self.player_state["ammo"] = self.PLAYER_AMMO_MAX # Refill ammo

            if self.player_state["health"] <= 0:
                reward -= 10
                self.game_over = True
                terminated = True
                # Sound: Player death
                self._create_particles(self.player_state["pos"], 50, self.COLOR_PLAYER, 3, 8)

            if self.steps >= self.MAX_STEPS:
                self.game_over = True
                terminated = True
        
        self.steps += 1
        self.last_action = action

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_player(self, movement, space_pressed, shift_pressed):
        p = self.player_state
        
        # Horizontal Movement
        if movement == 3: # Left
            p["pos"].x -= self.PLAYER_SPEED
            p["facing_right"] = False
        elif movement == 4: # Right
            p["pos"].x += self.PLAYER_SPEED
            p["facing_right"] = True

        # Clamp position to world bounds
        p["pos"].x = max(20, min(self.WORLD_WIDTH - 20, p["pos"].x))

        # Cooldowns
        if p["melee_cooldown"] > 0: p["melee_cooldown"] -= 1
        if p["ranged_cooldown"] > 0: p["ranged_cooldown"] -= 1
        if p["invuln_timer"] > 0: p["invuln_timer"] -= 1

        # Melee Attack
        if space_pressed and p["melee_cooldown"] == 0:
            p["melee_cooldown"] = 30 # 1 second cooldown
            direction = 1 if p["facing_right"] else -1
            self.melee_attack = {
                "rect": pygame.Rect(p["pos"].x + direction * 20, p["pos"].y - 30, 60, 40),
                "timer": 5, # Active for 5 frames
                "direction": direction,
                "dealt_damage": set()
            }
            # Sound: Melee swing
            
        # Ranged Attack
        if shift_pressed and p["ranged_cooldown"] == 0 and p["ammo"] > 0:
            p["ranged_cooldown"] = 15 # 0.5 second cooldown
            p["ammo"] -= 1
            direction = pygame.Vector2(1, 0) if p["facing_right"] else pygame.Vector2(-1, 0)
            start_pos = p["pos"] + direction * 25
            self.player_projectiles.append({"pos": start_pos, "vel": direction * 15})
            # Sound: Ranged fire
            self._create_particles(start_pos, 5, self.COLOR_AMMO, 1, 3, direction * -3)

        return 0

    def _update_monsters(self):
        reward = 0
        for m in self.monsters:
            player_dist = self.player_state["pos"].x - m["pos"].x
            
            # State transitions
            if abs(player_dist) < m["attack_range"] and m["attack_cooldown"] == 0:
                m["state"] = "attacking"
                m["vel"].x = 0
            elif abs(player_dist) < m["sight_range"]:
                m["state"] = "chasing"
            else:
                m["state"] = "patrolling"

            # Actions based on state
            if m["state"] == "attacking":
                m["attack_cooldown"] = m["attack_speed"]
                if m["type"] == "ranged":
                    # Sound: Monster ranged attack
                    dir_to_player = (self.player_state["pos"] - m["pos"]).normalize()
                    self.monster_projectiles.append({"pos": m["pos"].copy(), "vel": dir_to_player * 8})
                # Melee monster attack is handled by collision check
            elif m["state"] == "chasing":
                m["vel"].x = math.copysign(m["speed"], player_dist)
            elif m["state"] == "patrolling":
                if m["patrol_timer"] <= 0:
                    m["patrol_dir"] *= -1
                    m["patrol_timer"] = self.np_random.integers(90, 180)
                m["vel"].x = m["patrol_dir"] * m["speed"] * 0.5
                m["patrol_timer"] -= 1

            m["pos"] += m["vel"]
            m["pos"].x = max(0, min(self.WORLD_WIDTH, m["pos"].x))
            if m["attack_cooldown"] > 0: m["attack_cooldown"] -= 1
        return reward

    def _update_projectiles(self):
        for proj in self.player_projectiles[:]:
            proj["pos"] += proj["vel"]
            if not (0 < proj["pos"].x < self.WORLD_WIDTH):
                self.player_projectiles.remove(proj)
        for proj in self.monster_projectiles[:]:
            proj["pos"] += proj["vel"]
            if not (0 < proj["pos"].x < self.WORLD_WIDTH and 0 < proj["pos"].y < self.SCREEN_HEIGHT):
                self.monster_projectiles.remove(proj)
    
    def _update_effects(self):
        if self.screen_shake > 0: self.screen_shake -= 1
        self.camera_x += (self.player_state["pos"].x - self.SCREEN_WIDTH/2 - self.camera_x) * 0.1
        self.camera_x = max(0, min(self.WORLD_WIDTH - self.SCREEN_WIDTH, self.camera_x))

        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0: self.particles.remove(p)
        
        for d in self.damage_popups[:]:
            d["pos"].y -= 0.5
            d["life"] -= 1
            if d["life"] <= 0: self.damage_popups.remove(d)

        if self.melee_attack:
            self.melee_attack["timer"] -= 1
            if self.melee_attack["timer"] <= 0:
                self.melee_attack = None

    def _handle_collisions(self):
        reward = 0
        p_rect = pygame.Rect(self.player_state["pos"].x - 10, self.player_state["pos"].y - 25, 20, 50)

        # Player projectiles vs Monsters
        for proj in self.player_projectiles[:]:
            for m in self.monsters:
                m_rect = pygame.Rect(m["pos"].x - m["size"]/2, m["pos"].y - m["size"], m["size"], m["size"])
                if m_rect.collidepoint(proj["pos"]):
                    reward += self._damage_monster(m, 5) # Ranged damage
                    reward += 0.05
                    if proj in self.player_projectiles: self.player_projectiles.remove(proj)
                    break
        
        # Player melee vs Monsters
        if self.melee_attack:
            for i, m in enumerate(self.monsters):
                if i in self.melee_attack["dealt_damage"]: continue
                m_rect = pygame.Rect(m["pos"].x - m["size"]/2, m["pos"].y - m["size"], m["size"], m["size"])
                if self.melee_attack["rect"].colliderect(m_rect):
                    reward += self._damage_monster(m, 10) # Melee damage
                    reward += 0.1
                    self.melee_attack["dealt_damage"].add(i)

        # Monster projectiles vs Player
        for proj in self.monster_projectiles[:]:
            if p_rect.collidepoint(proj["pos"]):
                reward += self._damage_player(5)
                if proj in self.monster_projectiles: self.monster_projectiles.remove(proj)

        # Monster bodies vs Player (melee)
        for m in self.monsters:
            if m["type"] in ["slow_melee", "fast_melee"]:
                m_rect = pygame.Rect(m["pos"].x - m["size"]/2, m["pos"].y - m["size"], m["size"], m["size"])
                if p_rect.colliderect(m_rect):
                    reward += self._damage_player(5)

        return reward
    
    def _damage_player(self, amount):
        if self.player_state["invuln_timer"] == 0:
            self.player_state["health"] -= amount
            self.player_state["invuln_timer"] = 60 # 2 seconds invulnerability
            self.screen_shake = 10
            # Sound: Player hurt
            self._create_particles(self.player_state["pos"], 20, (255, 0, 0), 2, 4)
            return -0.02
        return 0.01 # Reward for dodging while invulnerable

    def _damage_monster(self, monster, amount):
        monster["health"] -= amount
        # Sound: Monster hurt
        self._create_particles(monster["pos"], 10, self.COLOR_DAMAGE, 1, 3)
        self.damage_popups.append({
            "pos": monster["pos"].copy() + pygame.Vector2(self.np_random.uniform(-10, 10), -monster["size"]),
            "text": str(amount),
            "life": 30
        })

        if monster["health"] <= 0:
            self.score += 100
            if monster in self.monsters: self.monsters.remove(monster)
            # Sound: Monster death
            self._create_particles(monster["pos"], 30, monster["color"], 2, 6)
            return 1.0 # Defeat monster bonus
        return 0

    def _spawn_wave(self):
        self.wave += 1
        num_monsters = 5
        for _ in range(num_monsters):
            monster_type = self.np_random.choice(["slow_melee", "fast_melee", "ranged"])
            side = 1 if self.np_random.random() < 0.5 else -1
            spawn_x = self.player_state["pos"].x + side * self.np_random.uniform(self.SCREEN_WIDTH/2, self.SCREEN_WIDTH/2 + 200)
            self._spawn_monster(monster_type, pygame.Vector2(spawn_x, self.SCREEN_HEIGHT - 50))
    
    def _spawn_monster(self, m_type, pos):
        base_health = 20
        base_speed = 1.5
        difficulty_mod = 1 + (self.wave - 1) * 0.05
        
        monster = {
            "type": m_type,
            "pos": pos,
            "vel": pygame.Vector2(0, 0),
            "health": base_health * difficulty_mod,
            "max_health": base_health * difficulty_mod,
            "state": "patrolling",
            "patrol_dir": 1 if self.np_random.random() < 0.5 else -1,
            "patrol_timer": self.np_random.integers(60, 120),
            "attack_cooldown": 0,
        }

        if m_type == "slow_melee":
            monster.update({"speed": base_speed * 0.8 * difficulty_mod, "size": 40, "color": (180, 40, 40), "sight_range": 250, "attack_range": 40, "attack_speed": 120})
        elif m_type == "fast_melee":
            monster.update({"speed": base_speed * 1.5 * difficulty_mod, "size": 25, "color": (255, 120, 0), "sight_range": 350, "attack_range": 30, "attack_speed": 75})
        elif m_type == "ranged":
            monster.update({"speed": base_speed * 1.0 * difficulty_mod, "size": 30, "color": (150, 50, 200), "sight_range": 400, "attack_range": 350, "attack_speed": 90})
        
        self.monsters.append(monster)

    def _create_particles(self, pos, count, color, min_speed, max_speed, initial_vel=pygame.Vector2(0,0)):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(min_speed, max_speed)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed + initial_vel
            self.particles.append({"pos": pos.copy(), "vel": vel, "life": self.np_random.integers(10, 20), "color": color})
    
    def _get_observation(self):
        # Camera offset for rendering
        cam_offset = pygame.Vector2(self.camera_x, 0)
        if self.screen_shake > 0:
            cam_offset.x += self.np_random.uniform(-5, 5)
            cam_offset.y += self.np_random.uniform(-5, 5)

        # --- Rendering ---
        self.screen.fill(self.COLOR_BG)
        self._render_background(cam_offset)
        self._render_entities(cam_offset)
        self._render_effects(cam_offset)
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self, cam_offset):
        # Parallax mountains
        for i in range(5):
            x = (i * 300 - cam_offset.x * 0.2) % (5 * 300) - 300
            pygame.gfxdraw.filled_trigon(self.screen, int(x), 350, int(x+150), 100, int(x+300), 350, (25, 30, 45))
        # Parallax hills
        for i in range(8):
            x = (i * 200 - cam_offset.x * 0.5) % (8 * 200) - 200
            pygame.draw.ellipse(self.screen, (35, 40, 55), (x, 250, 200, 200))
        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.SCREEN_HEIGHT - 50, self.SCREEN_WIDTH, 50))

    def _render_entities(self, cam_offset):
        p = self.player_state
        
        # Monsters
        for m in self.monsters:
            mx, my = (m["pos"] - cam_offset).xy
            size = m["size"]
            # Body
            body_rect = pygame.Rect(mx - size/2, my - size, size, size)
            pygame.draw.rect(self.screen, m["color"], body_rect)
            # Health bar
            health_pct = m["health"] / m["max_health"]
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (mx - size/2, my - size - 10, size, 5))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_ENEMY, (mx - size/2, my - size - 10, size * health_pct, 5))

        # Projectiles
        for proj in self.player_projectiles:
            px, py = (proj["pos"] - cam_offset).xy
            pygame.draw.line(self.screen, self.COLOR_DAMAGE, (px, py), (px - proj["vel"].x*0.5, py - proj["vel"].y*0.5), 4)
        for proj in self.monster_projectiles:
            px, py = (proj["pos"] - cam_offset).xy
            pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), 5, (255, 0, 150))

        # Player
        px, py = (p["pos"] - cam_offset).xy
        if p["invuln_timer"] % 10 < 5:
            # Glow effect
            glow_radius = 30
            s = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(s, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(s, (px - glow_radius, py - glow_radius - 15), special_flags=pygame.BLEND_RGBA_ADD)
            # Body
            player_rect = (px - 10, py - 25, 20, 50)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
            # Eye
            eye_x = px + (3 if p["facing_right"] else -7)
            pygame.draw.rect(self.screen, self.COLOR_BG, (eye_x, py-15, 4, 4))
    
    def _render_effects(self, cam_offset):
        # Melee swing
        if self.melee_attack:
            p_pos = self.player_state["pos"] - cam_offset
            angle_start = -math.pi/4 if self.melee_attack["direction"] > 0 else 5*math.pi/4
            angle_end = math.pi/4 if self.melee_attack["direction"] > 0 else 3*math.pi/4
            alpha = int(200 * (self.melee_attack["timer"] / 5))
            
            s = pygame.Surface((120, 120), pygame.SRCALPHA)
            pygame.draw.arc(s, (255,255,255,alpha), s.get_rect().inflate(-10,-10), angle_start, angle_end, 10)
            self.screen.blit(s, (p_pos.x-60, p_pos.y-60))

        # Particles
        for part in self.particles:
            pos = part["pos"] - cam_offset
            size = int(part["life"] * 0.2)
            pygame.draw.rect(self.screen, part["color"], (pos.x, pos.y, size, size))

        # Damage Popups
        for d in self.damage_popups:
            pos = d["pos"] - cam_offset
            alpha = int(255 * (d["life"] / 30))
            text_surf = self.font_small.render(d["text"], True, self.COLOR_DAMAGE)
            text_surf.set_alpha(alpha)
            self.screen.blit(text_surf, (pos.x - text_surf.get_width()/2, pos.y))

    def _render_ui(self):
        # Health Bar
        health_pct = self.player_state["health"] / self.PLAYER_HEALTH_MAX
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_PLAYER, (10, 10, 200 * health_pct, 20))
        
        # Ammo Bar
        ammo_pct = self.player_state["ammo"] / self.PLAYER_AMMO_MAX
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (10, 35, 200, 10))
        pygame.draw.rect(self.screen, self.COLOR_AMMO, (10, 35, 200 * ammo_pct, 10))

        # Score and Wave
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        wave_text = self.font_small.render(f"WAVE: {self.wave}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 40))

        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            end_text = self.font_large.render("GAME OVER", True, self.COLOR_HEALTH_ENEMY)
            self.screen.blit(end_text, (self.SCREEN_WIDTH/2 - end_text.get_width()/2, self.SCREEN_HEIGHT/2 - end_text.get_height()/2))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "player_health": self.player_state["health"],
            "player_ammo": self.player_state["ammo"],
            "monsters_left": len(self.monsters),
        }

    def close(self):
        pygame.font.quit()
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To run and play the game
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Human Controls ---
    # Map keyboard keys to the MultiDiscrete action space
    key_map = {
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
        pygame.K_SPACE: 1, # Corresponds to action[1]
        pygame.K_LSHIFT: 2, # Corresponds to action[2]
        pygame.K_RSHIFT: 2,
    }
    
    # Pygame window for human play
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption(env.game_description)
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not done:
        # --- Event Handling for Human Play ---
        action = np.array([0, 0, 0]) # Default action: no-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        
        # Movement (only one direction at a time)
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # Buttons (can be held simultaneously)
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
            
        # --- Step the Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Render to Screen ---
        # The observation is already a rendered frame, we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a bit before closing
            pygame.time.wait(2000)
            done = True
            
        clock.tick(30) # Run at 30 FPS
        
    env.close()