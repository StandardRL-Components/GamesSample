import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set Pygame to run in a headless mode
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move, ↑ to jump, ↓ to block. "
        "Space for a basic punch. "
        "Shift+Direction for special moves."
    )

    game_description = (
        "Two robots battle in a high-tech arena. Use punches and powerful special moves to "
        "defeat your opponent in a fast-paced duel."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_FLOOR = (40, 50, 80)
    COLOR_GRID = (30, 40, 65)
    
    COLOR_P1 = (255, 80, 80)
    COLOR_P1_ACCENT = (255, 150, 150)
    COLOR_P1_EYE = (255, 255, 255)
    
    COLOR_P2 = (80, 120, 255)
    COLOR_P2_ACCENT = (150, 180, 255)
    COLOR_P2_EYE = (255, 255, 255)

    COLOR_HEALTH_FG = (100, 220, 100)
    COLOR_HEALTH_BG = (80, 80, 80)
    COLOR_SPECIAL_READY = (255, 220, 0)
    COLOR_SPECIAL_CD = (80, 70, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_PUNCH_FX = (255, 255, 255)
    COLOR_BOLT_FX = (255, 255, 100)
    COLOR_SLAM_FX = (255, 120, 50)
    
    # Physics & Game Rules
    GRAVITY = 1.2
    PLAYER_SPEED = 6
    JUMP_VELOCITY = -16
    MAX_HEALTH = 100
    MAX_STEPS = 1500 # Increased for longer fights
    
    PUNCH_REACH = 40
    PUNCH_DURATION = 6 # frames
    PUNCH_DAMAGE = 5
    
    SPECIAL_COOLDOWN = [120, 150, 180] # Frames for each special
    
    # Special 1: Skyward Strike
    SKY_STRIKE_DAMAGE = 15
    SKY_STRIKE_VEL = -20

    # Special 2: Bolt Shot
    BOLT_DAMAGE = 10
    BOLT_SPEED = 12

    # Special 3: Ground Slam
    SLAM_DAMAGE = 20
    SLAM_RADIUS = 80
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 64)
        
        self.np_random = None
        
        # self.reset() is called by the wrapper/test harness, no need to call it here.

    def _create_robot_state(self, x_pos, color, accent_color, eye_color):
        return {
            "pos": pygame.Vector2(x_pos, self.HEIGHT - 50),
            "vel": pygame.Vector2(0, 0),
            "health": self.MAX_HEALTH,
            "on_ground": False,
            "is_blocking": False,
            "is_punching": 0, # Cooldown timer
            "is_hurt": 0, # Cooldown timer
            "action_state": "idle", # idle, jump, punch, hurt, block, special_1, special_2, special_3
            "facing_right": x_pos < self.WIDTH / 2,
            "specials_cooldown": [0, 0, 0],
            "colors": {"body": color, "accent": accent_color, "eye": eye_color}
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None or seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.player = self._create_robot_state(self.WIDTH // 4, self.COLOR_P1, self.COLOR_P1_ACCENT, self.COLOR_P1_EYE)
        self.opponent = self._create_robot_state(self.WIDTH * 3 // 4, self.COLOR_P2, self.COLOR_P2_ACCENT, self.COLOR_P2_EYE)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.winner = None

        self.projectiles = []
        self.particles = []
        self.screen_shake = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.ai_cooldown = 0
        self.ai_attack_frequency = 0.1

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held

        self._handle_player_input(movement, space_press, shift_press, shift_held)
        self._update_ai()
        
        self._update_physics(self.player)
        self._update_physics(self.opponent)

        combat_reward = self._handle_combat()
        reward += combat_reward

        self._update_cooldowns()
        self._update_particles()
        
        self.player["facing_right"] = self.player["pos"].x < self.opponent["pos"].x
        self.opponent["facing_right"] = self.opponent["pos"].x < self.player["pos"].x

        self.steps += 1
        self.ai_attack_frequency = min(0.8, 0.1 + (self.steps / 100) * 0.05)
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        if self.player["health"] <= 0:
            terminated = True
            self.game_over = True
            self.winner = "Opponent"
            reward -= 100
        elif self.opponent["health"] <= 0:
            terminated = True
            self.game_over = True
            self.winner = "Player"
            reward += 100
        elif self.steps >= self.MAX_STEPS:
            terminated = True # Use terminated for game-end conditions
            self.game_over = True
            if self.player["health"] > self.opponent["health"]:
                reward += 20
            elif self.opponent["health"] > self.player["health"]:
                reward -= 20
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
        
    def _handle_player_input(self, movement, space_press, shift_press, shift_held):
        p = self.player
        if p["is_hurt"] > 0 or self.game_over: return

        # Reset blocking state
        p["is_blocking"] = False

        # --- Special Moves (Highest Priority) ---
        if shift_held and p["action_state"] in ["idle", "walk", "jump_fall"]:
            # Special 1: Skyward Strike
            if movement == 1 and p["specials_cooldown"][0] <= 0:
                p["action_state"] = "special_1"
                p["vel"].y = self.SKY_STRIKE_VEL
                p["specials_cooldown"][0] = self.SPECIAL_COOLDOWN[0]
                # SFX: whoosh_up
                self._create_particles(p["pos"], 15, self.COLOR_P1_ACCENT, 2, 8, -math.pi/2, math.pi/4)

            # Special 2: Bolt Shot
            elif movement in [3, 4] and p["specials_cooldown"][1] <= 0:
                p["action_state"] = "special_2"
                p["is_punching"] = self.PUNCH_DURATION # Use punch anim
                p["specials_cooldown"][1] = self.SPECIAL_COOLDOWN[1]
                bolt_dir = 1 if p["facing_right"] else -1
                start_pos = p["pos"] + pygame.Vector2(bolt_dir * 30, -20)
                self.projectiles.append({
                    "pos": start_pos, "vel": pygame.Vector2(bolt_dir * self.BOLT_SPEED, 0),
                    "owner": self.player, "color": self.COLOR_BOLT_FX
                })
                # SFX: laser_shoot
            
            # Special 3: Ground Slam
            elif movement == 2 and p["specials_cooldown"][2] <= 0 and p["on_ground"]:
                p["action_state"] = "special_3"
                p["vel"].y = -8 # Small hop
                p["specials_cooldown"][2] = self.SPECIAL_COOLDOWN[2]
                # SFX: charge_up

        # --- Basic Attack ---
        elif space_press and p["is_punching"] <= 0 and p["action_state"] not in ["special_1", "special_2", "special_3"]:
            p["is_punching"] = self.PUNCH_DURATION
            p["action_state"] = "punch"
            # SFX: punch_swing

        # --- Movement & Blocking ---
        elif not p["is_punching"] > 0 and p["action_state"] not in ["special_1", "special_3"]:
            # Horizontal Movement
            if movement == 3: # Left
                p["vel"].x = -self.PLAYER_SPEED
                p["action_state"] = "walk"
            elif movement == 4: # Right
                p["vel"].x = self.PLAYER_SPEED
                p["action_state"] = "walk"
            else:
                p["vel"].x = 0
                if p["on_ground"]: p["action_state"] = "idle"

            # Jump
            if movement == 1 and p["on_ground"]:
                p["vel"].y = self.JUMP_VELOCITY
                p["on_ground"] = False
                p["action_state"] = "jump_fall"
                # SFX: jump
                self._create_particles(p["pos"] + pygame.Vector2(0, 10), 10, (200,200,200), 1, 3, math.pi, math.pi/2)

            # Block
            if movement == 2 and p["on_ground"]:
                p["is_blocking"] = True
                p["action_state"] = "block"
                p["vel"].x = 0

    def _update_ai(self):
        p, o = self.player, self.opponent
        if o["is_hurt"] > 0 or self.game_over: return
        
        o["is_blocking"] = False
        dist = p["pos"].distance_to(o["pos"])
        
        if self.ai_cooldown <= 0:
            self.ai_cooldown = self.np_random.integers(10, 25)
            
            # Decide action
            rand_val = self.np_random.random()
            
            # Use special moves
            can_use_special = any(cd <= 0 for cd in o["specials_cooldown"])
            if can_use_special and rand_val < 0.1:
                # Try to use a ready special
                ready_specials = [i for i, cd in enumerate(o["specials_cooldown"]) if cd <= 0]
                choice = self.np_random.choice(ready_specials)

                if choice == 0: # Sky Strike (use if player is above)
                    if p["pos"].y < o["pos"].y - 30 and abs(p["pos"].x - o["pos"].x) < 50:
                        o["action_state"] = "special_1"
                        o["vel"].y = self.SKY_STRIKE_VEL
                        o["specials_cooldown"][0] = self.SPECIAL_COOLDOWN[0]
                elif choice == 1: # Bolt Shot (use at range)
                    if dist > 150:
                        o["action_state"] = "special_2"
                        o["is_punching"] = self.PUNCH_DURATION
                        o["specials_cooldown"][1] = self.SPECIAL_COOLDOWN[1]
                        bolt_dir = 1 if o["facing_right"] else -1
                        start_pos = o["pos"] + pygame.Vector2(bolt_dir * 30, -20)
                        self.projectiles.append({
                            "pos": start_pos, "vel": pygame.Vector2(bolt_dir * self.BOLT_SPEED, 0),
                            "owner": self.opponent, "color": self.COLOR_BOLT_FX
                        })
                elif choice == 2: # Ground Slam (use if player is close)
                    if dist < self.SLAM_RADIUS and o["on_ground"]:
                         o["action_state"] = "special_3"
                         o["vel"].y = -8
                         o["specials_cooldown"][2] = self.SPECIAL_COOLDOWN[2]
                return

            # Basic punch
            if dist < self.PUNCH_REACH + 20 and rand_val < self.ai_attack_frequency:
                o["is_punching"] = self.PUNCH_DURATION
                o["action_state"] = "punch"
                o["vel"].x = 0
            # Move towards player
            elif dist > self.PUNCH_REACH:
                o["vel"].x = self.PLAYER_SPEED * (1 if p["pos"].x > o["pos"].x else -1)
                o["action_state"] = "walk"
            # Move away
            elif dist < self.PUNCH_REACH - 10:
                o["vel"].x = self.PLAYER_SPEED * (-1 if p["pos"].x > o["pos"].x else 1)
                o["action_state"] = "walk"
            else:
                o["vel"].x = 0
                o["action_state"] = "idle"
        else:
            # Maintain movement if not attacking
            if o["is_punching"] <= 0 and o["action_state"] not in ["special_1", "special_3"]:
                if o["action_state"] == "walk":
                    pass # Keep moving
                else:
                    o["vel"].x = 0


    def _update_physics(self, entity):
        # Apply gravity
        if not entity["on_ground"]:
            entity["vel"].y += self.GRAVITY
        
        # Move
        entity["pos"] += entity["vel"]

        # Ground collision
        floor_y = self.HEIGHT - 50
        if entity["pos"].y >= floor_y:
            entity["pos"].y = floor_y
            entity["vel"].y = 0
            if not entity["on_ground"]: # Landed
                # SFX: land
                self._create_particles(entity["pos"] + pygame.Vector2(0, 10), 5, (200,200,200), 1, 2, math.pi, math.pi/2)
                
                # Ground Slam effect
                if entity["action_state"] == "special_3":
                    self.screen_shake = 10
                    # SFX: explosion
                    self._create_particles(entity["pos"], 40, self.COLOR_SLAM_FX, 3, 10, 0, 2*math.pi)

            entity["on_ground"] = True
            if entity["action_state"] in ["jump_fall", "special_1", "special_3"]:
                entity["action_state"] = "idle"
        
        # Wall collisions
        entity["pos"].x = max(20, min(self.WIDTH - 20, entity["pos"].x))

    def _handle_combat(self):
        reward = 0
        entities = [self.player, self.opponent]
        
        # --- Handle Active Attacks (Punch, Sky Strike) ---
        for i, attacker in enumerate(entities):
            defender = entities[1-i]
            
            # Basic Punch
            if attacker["is_punching"] > 0 and attacker["action_state"] == "punch":
                direction = 1 if attacker["facing_right"] else -1
                punch_rect = pygame.Rect(
                    attacker["pos"].x if direction == 1 else attacker["pos"].x - self.PUNCH_REACH,
                    attacker["pos"].y - 40,
                    self.PUNCH_REACH, 40
                )
                defender_rect = pygame.Rect(defender["pos"].x - 15, defender["pos"].y - 50, 30, 60)

                if punch_rect.colliderect(defender_rect):
                    if defender["is_blocking"]:
                        # SFX: block
                        reward += -0.5 if attacker == self.player else 0
                        self._create_particles(defender["pos"] + pygame.Vector2(0, -25), 5, (200, 200, 255), 1, 3)
                    else:
                        # SFX: hit_impact
                        reward += 0.1 if attacker == self.player else -0.1
                        defender["health"] = max(0, defender["health"] - self.PUNCH_DAMAGE)
                        defender["is_hurt"] = 10
                        defender["vel"].x += direction * 5
                        defender["vel"].y = -3
                        self._create_particles(pygame.Vector2(punch_rect.center), 10, self.COLOR_PUNCH_FX, 2, 5)
                        self.screen_shake = 5
                    attacker["is_punching"] = 0 # Hit connects, end attack

            # Skyward Strike
            if attacker["action_state"] == "special_1":
                attacker_rect = pygame.Rect(attacker["pos"].x - 15, attacker["pos"].y - 60, 30, 70)
                defender_rect = pygame.Rect(defender["pos"].x - 15, defender["pos"].y - 50, 30, 60)
                if attacker_rect.colliderect(defender_rect) and not defender["is_blocking"]:
                    # SFX: special_hit
                    reward += 1.0 if attacker == self.player else -0.1 # High reward for player, normal penalty
                    defender["health"] = max(0, defender["health"] - self.SKY_STRIKE_DAMAGE)
                    defender["is_hurt"] = 15
                    defender["vel"].y = -15
                    defender["vel"].x += (1 if attacker["facing_right"] else -1) * 8
                    self.screen_shake = 8
                    attacker["action_state"] = "jump_fall" # End the attack part
                    self._create_particles(defender["pos"], 20, attacker["colors"]["accent"], 3, 6)

            # Ground Slam Damage
            if attacker["action_state"] == "idle" and attacker["on_ground"] and self.screen_shake > 0 and attacker["specials_cooldown"][2] > self.SPECIAL_COOLDOWN[2] - 10:
                if attacker["pos"].distance_to(defender["pos"]) < self.SLAM_RADIUS and not defender["is_blocking"]:
                    # SFX: special_hit_heavy
                    reward += 1.0 if attacker == self.player else -0.1
                    defender["health"] = max(0, defender["health"] - self.SLAM_DAMAGE)
                    defender["is_hurt"] = 20
                    defender["vel"].y = -12
                    dx = defender["pos"].x - attacker["pos"].x
                    direction = 1 if dx > 0 else -1
                    defender["vel"].x += direction * 10
                    
        # --- Handle Projectiles ---
        for proj in self.projectiles[:]:
            proj["pos"] += proj["vel"]
            
            target = self.opponent if proj["owner"] == self.player else self.player
            target_rect = pygame.Rect(target["pos"].x - 15, target["pos"].y - 50, 30, 60)
            
            if target_rect.collidepoint(proj["pos"]):
                if target["is_blocking"]:
                    # SFX: block_energy
                    reward += -0.5 if proj["owner"] == self.player else 0
                else:
                    # SFX: hit_energy
                    reward += 1.0 if proj["owner"] == self.player else -0.1
                    target["health"] = max(0, target["health"] - self.BOLT_DAMAGE)
                    target["is_hurt"] = 12
                    target["vel"].x += proj["vel"].x * 0.5
                self.projectiles.remove(proj)
                self._create_particles(proj["pos"], 15, proj["color"], 2, 4)
                self.screen_shake = 6
            elif not (0 < proj["pos"].x < self.WIDTH):
                self.projectiles.remove(proj)
                
        return reward
        
    def _update_cooldowns(self):
        for entity in [self.player, self.opponent]:
            if entity["is_punching"] > 0: entity["is_punching"] -= 1
            if entity["is_hurt"] > 0: entity["is_hurt"] -= 1
            for i in range(3):
                if entity["specials_cooldown"][i] > 0:
                    entity["specials_cooldown"][i] -= 1
        if self.ai_cooldown > 0: self.ai_cooldown -= 1
        if self.screen_shake > 0: self.screen_shake -= 1

    def _create_particles(self, pos, count, color, min_speed, max_speed, angle=0, spread=2*math.pi):
        for _ in range(count):
            p_angle = angle + self.np_random.uniform(-spread/2, spread/2)
            p_speed = self.np_random.uniform(min_speed, max_speed)
            vel = pygame.Vector2(math.cos(p_angle), math.sin(p_angle)) * p_speed
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "lifespan": self.np_random.integers(10, 20),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.9 # friction
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        # Apply screen shake
        render_offset = pygame.Vector2(0, 0)
        if self.screen_shake > 0:
            render_offset.x = self.np_random.integers(-self.screen_shake, self.screen_shake + 1)
            render_offset.y = self.np_random.integers(-self.screen_shake, self.screen_shake + 1)

        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        self._render_background(render_offset)
        self._render_game(render_offset)
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self, offset):
        # Parallax stars
        for i in range(50):
            # Use step and index to create a deterministic but moving seed
            star_seed = (i * 13 + self.steps // 4) % (self.WIDTH * self.HEIGHT)
            x = star_seed % self.WIDTH
            y = (star_seed * 31) % self.HEIGHT
            pygame.draw.rect(self.screen, self.COLOR_GRID, (x + offset.x, y + offset.y, 1, 1))

        # Grid Floor
        floor_y = self.HEIGHT - 50
        pygame.draw.rect(self.screen, self.COLOR_FLOOR, (0 + offset.x, floor_y + offset.y, self.WIDTH, 50))
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i + offset.x, floor_y + offset.y), (i-20 + offset.x, self.HEIGHT + offset.y), 2)
    
    def _render_robot(self, entity, offset):
        pos = entity["pos"] + offset
        colors = entity["colors"]
        
        # Hurt flash
        if entity["is_hurt"] > 0 and (self.steps // 2) % 2 == 0:
            hurt_surf = pygame.Surface((50, 60), pygame.SRCALPHA)
            hurt_surf.fill((255, 100, 100, 150))
            self.screen.blit(hurt_surf, (pos.x - 25, pos.y - 55))

        # Body parts
        body_h = 40
        body_w = 30
        head_r = 15
        
        # Animation squish and stretch
        squish = 0
        if entity["action_state"] == "block": squish = 10
        elif entity["is_punching"] > 0: squish = -5
        
        # Body
        body_rect = pygame.Rect(pos.x - body_w/2, pos.y - body_h - squish, body_w, body_h + squish)
        pygame.draw.rect(self.screen, colors["body"], body_rect, border_radius=5)
        
        # Head
        head_pos = (int(pos.x), int(pos.y - body_h - head_r + squish / 2))
        pygame.draw.circle(self.screen, colors["accent"], head_pos, head_r)
        
        # Eye
        eye_dir = 1 if entity["facing_right"] else -1
        eye_pos = (head_pos[0] + eye_dir * 7, head_pos[1])
        pygame.draw.circle(self.screen, colors["eye"], eye_pos, 4)
        
        # Punch effect
        if entity["is_punching"] > 0 and entity["action_state"] == "punch":
            reach = self.PUNCH_REACH * (1 - entity["is_punching"] / self.PUNCH_DURATION)
            punch_start = pos + pygame.Vector2(eye_dir * 15, -30)
            punch_end = punch_start + pygame.Vector2(eye_dir * reach, 0)
            pygame.draw.line(self.screen, self.COLOR_PUNCH_FX, punch_start, punch_end, 4)

        # Block effect
        if entity["is_blocking"]:
            block_surf = pygame.Surface((60, 70), pygame.SRCALPHA)
            block_surf.fill((150, 180, 255, 100))
            self.screen.blit(block_surf, (pos.x-30, pos.y-60))

    def _render_game(self, offset):
        self._render_robot(self.player, offset)
        self._render_robot(self.opponent, offset)

        for p in self.particles:
            size = max(1, p["lifespan"] / 4)
            pygame.draw.circle(self.screen, p["color"], p["pos"] + offset, size)
            
        for proj in self.projectiles:
            pygame.draw.circle(self.screen, proj["color"], proj["pos"] + offset, 5)
            # Add a glow
            pygame.gfxdraw.filled_circle(self.screen, int(proj["pos"].x + offset.x), int(proj["pos"].y + offset.y), 8, (*proj["color"], 100))
            
    def _render_ui(self):
        # Player Health
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (20, 20, 250, 20))
        p1_health_w = max(0, (self.player["health"] / self.MAX_HEALTH) * 250)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (20, 20, p1_health_w, 20))
        p1_text = self.font_small.render("PLAYER", True, self.COLOR_TEXT)
        self.screen.blit(p1_text, (25, 21))

        # Opponent Health
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (self.WIDTH - 270, 20, 250, 20))
        p2_health_w = max(0, (self.opponent["health"] / self.MAX_HEALTH) * 250)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (self.WIDTH - 270 + (250 - p2_health_w), 20, p2_health_w, 20))
        p2_text = self.font_small.render("OPPONENT", True, self.COLOR_TEXT)
        self.screen.blit(p2_text, (self.WIDTH - 265, 21))

        # Player Specials
        for i in range(3):
            color = self.COLOR_SPECIAL_READY if self.player["specials_cooldown"][i] <= 0 else self.COLOR_SPECIAL_CD
            pygame.draw.circle(self.screen, color, (30 + i * 25, 55), 8)
        
        # Opponent Specials
        for i in range(3):
            color = self.COLOR_SPECIAL_READY if self.opponent["specials_cooldown"][i] <= 0 else self.COLOR_SPECIAL_CD
            pygame.draw.circle(self.screen, color, (self.WIDTH - 30 - i * 25, 55), 8)

        # Timer
        time_left = max(0, self.MAX_STEPS - self.steps) // self.FPS
        time_text = self.font_large.render(str(time_left), True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH // 2 - time_text.get_width() // 2, 10))

        # Game Over Text
        if self.game_over:
            end_text_str = f"{self.winner.upper()} WINS!" if self.winner else "TIME UP"
            end_text = self.font_large.render(end_text_str, True, self.COLOR_SPECIAL_READY)
            self.screen.blit(end_text, (self.WIDTH // 2 - end_text.get_width() // 2, self.HEIGHT // 2 - 50))
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player["health"],
            "opponent_health": self.opponent["health"],
        }
        
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game with keyboard controls
    # It will open a window for human play
    os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a window for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Robot Battle Arena")
    
    # Key mapping
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while not done:
        # --- Human Controls ---
        movement = 0
        space = 0
        shift = 0
        
        keys = pygame.key.get_pressed()
        for key, move_val in key_map.items():
            if keys[key]:
                movement = move_val
                break
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = [movement, space, shift]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Pygame Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        env.clock.tick(env.FPS)
        
        if done:
            print(f"Game Over. Final Info: {info}")
            # Wait a bit before closing
            pygame.time.wait(2000)

    env.close()