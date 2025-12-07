import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:33:01.556053
# Source Brief: brief_00482.md
# Brief Index: 482
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Guide a single-celled organism on an evolutionary journey, climbing platforms and "
        "collecting DNA to unlock new abilities and reach the nucleus."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move and jump. Press space to extend a pseudopod "
        "to collect DNA. Use shift to activate a speed boost."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- CRITICAL: GYMNASIUM SPACES ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- PYGAME SETUP ---
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_ability = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_title = pygame.font.SysFont("monospace", 40, bold=True)

        # --- VISUALS & COLORS ---
        self.COLOR_BG = (15, 5, 25)
        self.COLOR_BG_ACCENT = (30, 10, 50)
        self.COLOR_PLAYER = (255, 200, 0)
        self.COLOR_PLAYER_GLOW = (255, 200, 0, 50)
        self.COLOR_PLATFORM = (0, 180, 120)
        self.COLOR_PLATFORM_GLOW = (0, 180, 120, 40)
        self.COLOR_DNA = (50, 150, 255)
        self.COLOR_DNA_GLOW = (50, 150, 255, 100)
        self.COLOR_NUCLEUS = (200, 0, 255)
        self.COLOR_NUCLEUS_GLOW = (200, 0, 255, 60)
        self.COLOR_PSEUDOPOD = (255, 255, 255)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_ABILITY_LOCKED = (100, 100, 100)
        self.COLOR_ABILITY_UNLOCKED = (200, 220, 255)

        # --- GAME CONSTANTS ---
        self.WORLD_HEIGHT = 4000
        self.MAX_STEPS = 5000
        self.GRAVITY = 0.4
        self.PLAYER_JUMP_STRENGTH = -9
        self.PLAYER_MOVE_SPEED = 4.0
        self.CILIA_BOOST_STRENGTH = 6.0
        self.CILIA_BOOST_DURATION = 10
        self.CILIA_BOOST_COOLDOWN_MAX = 60 # 2 seconds at 30fps
        self.PSEUDOPOD_MAX_LENGTH = 120
        self.PSEUDOPOD_SPEED = 20
        self.PSEUDOPOD_COOLDOWN_MAX = 20
        self.DNA_TO_UNLOCK_DOUBLE_JUMP = 5
        self.DNA_TO_UNLOCK_CILIA_BOOST = 10

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_radius = 12
        self.on_ground = False
        self.jumps_left = 0
        self.aim_direction = pygame.Vector2(0, 1)

        self.platforms = []
        self.dna_strands = []
        self.particles = []
        self.nucleus_pos = pygame.Vector2(0, 0)
        
        self.camera_y = 0.0
        
        self.dna_collected = 0
        self.unlocked_abilities = {"double_jump": False, "cilia_boost": False}
        self.cilia_boost_timer = 0
        self.cilia_boost_cooldown = 0
        
        self.pseudopod_state = "retracted" # retracted, extending, retracting
        self.pseudopod_length = 0
        self.pseudopod_cooldown = 0

        self.platform_speed_multiplier = 1.0
        self.last_reward_info = ""

        # --- INITIALIZE AND VALIDATE ---
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False

        # Player state
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.WORLD_HEIGHT - 50)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False
        self.jumps_left = 1
        self.aim_direction = pygame.Vector2(0, 1)

        # World generation
        self._generate_world()
        
        self.camera_y = self.player_pos.y - self.HEIGHT * 0.75
        
        # Progression state
        self.dna_collected = 0
        self.unlocked_abilities = {"double_jump": False, "cilia_boost": False}
        
        # Ability timers
        self.cilia_boost_timer = 0
        self.cilia_boost_cooldown = 0
        
        # Pseudopod state
        self.pseudopod_state = "retracted"
        self.pseudopod_length = 0
        self.pseudopod_cooldown = 0

        self.platform_speed_multiplier = 1.0
        self.particles.clear()
        
        self.last_reward_info = ""

        return self._get_observation(), self._get_info()

    def _generate_world(self):
        self.platforms.clear()
        self.dna_strands.clear()
        
        # Starting platform
        start_plat = pygame.Rect(self.WIDTH/2 - 75, self.WORLD_HEIGHT - 20, 150, 20)
        self.platforms.append({'rect': start_plat, 'move': None})
        
        last_y = start_plat.y
        last_x = start_plat.centerx
        
        # Procedurally generate platforms upwards
        while last_y > 200:
            width = self.np_random.integers(80, 150)
            height = 20
            
            # Ensure next platform is reachable
            offset_x = self.np_random.uniform(-150, 150)
            offset_y = self.np_random.uniform(80, 130) # Vertical gap
            
            new_x = np.clip(last_x + offset_x, width/2, self.WIDTH - width/2)
            new_y = last_y - offset_y
            
            rect = pygame.Rect(new_x - width/2, new_y, width, height)
            
            # Add movement pattern to some platforms
            move_pattern = None
            if self.np_random.random() < 0.6:
                if self.np_random.random() < 0.5: # Horizontal
                    move_pattern = {
                        'type': 'sin_x', 'center': rect.x, 'amp': self.np_random.uniform(30, 80),
                        'freq': self.np_random.uniform(0.01, 0.03)
                    }
                else: # Vertical
                     move_pattern = {
                        'type': 'sin_y', 'center': rect.y, 'amp': self.np_random.uniform(10, 40),
                        'freq': self.np_random.uniform(0.01, 0.03)
                    }
            self.platforms.append({'rect': rect, 'move': move_pattern})

            # Scatter DNA strands near platforms
            if self.np_random.random() < 0.5:
                dna_pos = pygame.Vector2(
                    rect.centerx + self.np_random.uniform(-80, 80),
                    rect.y - self.np_random.uniform(20, 60)
                )
                self.dna_strands.append({'pos': dna_pos, 'collected': False})

            last_y = new_y
            last_x = new_x
        
        # Place nucleus at the top
        self.nucleus_pos = pygame.Vector2(last_x, last_y - 150)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- UPDATE GAME LOGIC ---
        self._handle_input(movement, space_held, shift_held)
        reward += self._update_player_physics()
        self._update_platforms()
        self._update_pseudopod()
        self._update_particles()
        
        # --- HANDLE COLLISIONS AND REWARDS ---
        reward += self._handle_collisions()
        
        # Difficulty scaling
        if self.steps > 0 and self.steps % 250 == 0:
            self.platform_speed_multiplier += 0.05

        # Update camera
        self.camera_y += (self.player_pos.y - self.HEIGHT * 0.6 - self.camera_y) * 0.1
        self.camera_y = min(self.camera_y, self.WORLD_HEIGHT - self.HEIGHT)
        
        terminated = self._check_termination()
        
        # Terminal rewards
        if terminated:
            if self.player_pos.y > self.WORLD_HEIGHT:
                reward = -100.0 # Fell
                self.last_reward_info = "Fell into Cytoplasm (-100)"
            elif self.player_pos.distance_to(self.nucleus_pos) < self.player_radius + 40:
                reward = 100.0 # Reached nucleus
                self.last_reward_info = "Reached Nucleus! (+100)"

        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # --- Movement ---
        # 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1: # Up / Jump
            if self.on_ground:
                self.player_vel.y = self.PLAYER_JUMP_STRENGTH
                self.jumps_left = 1 if not self.unlocked_abilities["double_jump"] else 2
                self.jumps_left -= 1
                self.on_ground = False
                self._create_particles(self.player_pos + pygame.Vector2(0, self.player_radius), 10, self.COLOR_PLATFORM)
                # sfx: jump
            elif self.jumps_left > 0:
                self.player_vel.y = self.PLAYER_JUMP_STRENGTH * 0.9 # 2nd jump slightly weaker
                self.jumps_left -= 1
                self._create_particles(self.player_pos, 15, self.COLOR_ABILITY_UNLOCKED)
                # sfx: double_jump
            self.aim_direction = pygame.Vector2(0, -1)
        elif movement == 2: # Down
            self.player_vel.y = min(self.player_vel.y + 1, 10) # Fast fall
            self.aim_direction = pygame.Vector2(0, 1)
        elif movement == 3: # Left
            self.player_vel.x -= 1.0
            self.aim_direction = pygame.Vector2(-1, 0)
        elif movement == 4: # Right
            self.player_vel.x += 1.0
            self.aim_direction = pygame.Vector2(1, 0)
        
        # Update aim direction if moving diagonally
        if movement in [1, 2] and self.player_vel.x != 0:
            self.aim_direction.x = np.sign(self.player_vel.x)
            self.aim_direction.normalize_ip()

        # --- Special Ability (Shift) ---
        if shift_held and self.unlocked_abilities["cilia_boost"] and self.cilia_boost_cooldown == 0:
            self.cilia_boost_timer = self.CILIA_BOOST_DURATION
            self.cilia_boost_cooldown = self.CILIA_BOOST_COOLDOWN_MAX
            self._create_particles(self.player_pos, 20, (255, 100, 255))
            # sfx: boost_activate

        # --- Pseudopod (Space) ---
        if space_held and self.pseudopod_state == "retracted" and self.pseudopod_cooldown == 0:
            self.pseudopod_state = "extending"
            # sfx: pseudopod_extend

    def _update_player_physics(self):
        # Apply Cilia Boost
        if self.cilia_boost_timer > 0:
            boost_dir = pygame.Vector2(self.player_vel.x, 0).normalize() if self.player_vel.x != 0 else pygame.Vector2(1,0)
            self.player_vel += boost_dir * self.CILIA_BOOST_STRENGTH
            self.cilia_boost_timer -= 1

        # Apply gravity
        if not self.on_ground:
            self.player_vel.y += self.GRAVITY
        
        # Apply friction/drag
        self.player_vel.x *= 0.85 

        # Clamp velocity
        self.player_vel.x = np.clip(self.player_vel.x, -self.PLAYER_MOVE_SPEED, self.PLAYER_MOVE_SPEED)
        self.player_vel.y = min(self.player_vel.y, 15)

        # Update position
        self.player_pos += self.player_vel
        
        # Screen bounds
        if self.player_pos.x < self.player_radius:
            self.player_pos.x = self.player_radius
            self.player_vel.x *= -0.5
        if self.player_pos.x > self.WIDTH - self.player_radius:
            self.player_pos.x = self.WIDTH - self.player_radius
            self.player_vel.x *= -0.5
            
        # Cooldowns
        if self.cilia_boost_cooldown > 0: self.cilia_boost_cooldown -= 1
        if self.pseudopod_cooldown > 0: self.pseudopod_cooldown -= 1

        return 0.0

    def _update_platforms(self):
        for p in self.platforms:
            if p['move']:
                pattern = p['move']
                speed = self.platform_speed_multiplier
                if pattern['type'] == 'sin_x':
                    p['rect'].x = pattern['center'] + math.sin(self.steps * pattern['freq'] * speed) * pattern['amp']
                elif pattern['type'] == 'sin_y':
                    p['rect'].y = pattern['center'] + math.sin(self.steps * pattern['freq'] * speed) * pattern['amp']

    def _update_pseudopod(self):
        if self.pseudopod_state == "extending":
            self.pseudopod_length += self.PSEUDOPOD_SPEED
            if self.pseudopod_length >= self.PSEUDOPOD_MAX_LENGTH:
                self.pseudopod_length = self.PSEUDOPOD_MAX_LENGTH
                self.pseudopod_state = "retracting"
        elif self.pseudopod_state == "retracting":
            self.pseudopod_length -= self.PSEUDOPOD_SPEED
            if self.pseudopod_length <= 0:
                self.pseudopod_length = 0
                self.pseudopod_state = "retracted"
                self.pseudopod_cooldown = self.PSEUDOPOD_COOLDOWN_MAX

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] *= 0.97

    def _handle_collisions(self):
        reward = 0.0
        
        # Player-Platform collision
        self.on_ground = False
        player_rect = pygame.Rect(self.player_pos.x - self.player_radius, self.player_pos.y - self.player_radius, self.player_radius*2, self.player_radius*2)
        
        for p in self.platforms:
            if p['rect'].colliderect(player_rect):
                # Check if landing on top
                if self.player_vel.y > 0 and player_rect.bottom < p['rect'].centery:
                    self.player_pos.y = p['rect'].top - self.player_radius
                    self.player_vel.y = 0
                    self.on_ground = True
                    self.jumps_left = 1 if not self.unlocked_abilities["double_jump"] else 2
                    
        # Pseudopod-DNA collision
        if self.pseudopod_state == "extending":
            pod_end = self.player_pos + self.aim_direction * self.pseudopod_length
            for dna in self.dna_strands:
                if not dna['collected'] and dna['pos'].distance_to(pod_end) < 20:
                    dna['collected'] = True
                    self.dna_collected += 1
                    reward += 0.1
                    self.last_reward_info = "DNA Matched (+0.1)"
                    self._create_particles(dna['pos'], 30, self.COLOR_DNA)
                    # sfx: dna_collect
                    
                    # Check for ability unlocks
                    if not self.unlocked_abilities["double_jump"] and self.dna_collected >= self.DNA_TO_UNLOCK_DOUBLE_JUMP:
                        self.unlocked_abilities["double_jump"] = True
                        reward += 1.0
                        self.last_reward_info = "Double Jump Unlocked! (+1.0)"
                        # sfx: ability_unlock
                    if not self.unlocked_abilities["cilia_boost"] and self.dna_collected >= self.DNA_TO_UNLOCK_CILIA_BOOST:
                        self.unlocked_abilities["cilia_boost"] = True
                        reward += 1.0
                        self.last_reward_info = "Cilia Boost Unlocked! (+1.0)"
                        # sfx: ability_unlock
        return reward

    def _check_termination(self):
        # Fall out of world
        if self.player_pos.y > self.WORLD_HEIGHT:
            self.game_over = True
        # Reach nucleus
        if self.player_pos.distance_to(self.nucleus_pos) < self.player_radius + 40:
            self.game_over = True
        # Max steps
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        return self.game_over

    def _get_observation(self):
        # --- RENDER GAME ---
        # Background
        self.screen.fill(self.COLOR_BG)
        for i in range(10): # Parallax background cells
            size = 100 - i * 8
            offset_y = (self.camera_y * (i * 0.1)) % size
            pygame.gfxdraw.filled_circle(self.screen, (i*70)%self.WIDTH, int(100+i*30 + offset_y), int(size/2), self.COLOR_BG_ACCENT)

        # Game elements
        self._render_nucleus()
        self._render_platforms()
        self._render_dna()
        self._render_particles()
        self._render_player()
        self._render_pseudopod()
        
        # UI
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_player(self):
        pos = (int(self.player_pos.x), int(self.player_pos.y - self.camera_y))
        
        # Cilia boost effect
        if self.cilia_boost_timer > 0:
            for i in range(5):
                offset = (self.np_random.random() - 0.5) * self.player_radius
                alpha = 150 - i * 30
                pygame.gfxdraw.filled_circle(self.screen, pos[0] - int(self.player_vel.x * i * 0.5), pos[1] - int(self.player_vel.y * i * 0.5) + int(offset), self.player_radius, (255, 100, 255, alpha))

        # Glow
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.player_radius + 5, self.COLOR_PLAYER_GLOW)
        # Body
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.player_radius, self.COLOR_PLAYER)
        
        # Eye indicating aim direction
        eye_pos = (pos[0] + int(self.aim_direction.x * self.player_radius * 0.5),
                   pos[1] + int(self.aim_direction.y * self.player_radius * 0.5))
        pygame.gfxdraw.filled_circle(self.screen, eye_pos[0], eye_pos[1], 3, (255, 255, 255))

    def _render_platforms(self):
        for p in self.platforms:
            rect_on_cam = p['rect'].move(0, -self.camera_y)
            if rect_on_cam.bottom < 0 or rect_on_cam.top > self.HEIGHT:
                continue
            
            # Use rounded rects for organic feel
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_GLOW, rect_on_cam.inflate(10,10), border_radius=10)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, rect_on_cam, border_radius=8)

    def _render_dna(self):
        for dna in self.dna_strands:
            if not dna['collected']:
                pos = (int(dna['pos'].x), int(dna['pos'].y - self.camera_y + math.sin(self.steps * 0.05 + dna['pos'].x) * 5))
                if pos[1] < -20 or pos[1] > self.HEIGHT + 20:
                    continue

                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 12, self.COLOR_DNA_GLOW)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_DNA)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_DNA)

    def _render_nucleus(self):
        pos = (int(self.nucleus_pos.x), int(self.nucleus_pos.y - self.camera_y))
        if pos[1] < -100 or pos[1] > self.HEIGHT + 100:
            return

        radius = 40 + int(math.sin(self.steps * 0.02) * 5)
        # Pulsating glow
        for i in range(5):
            glow_rad = radius + i * 5 + int(math.sin(self.steps*0.03 + i)*5)
            alpha = 100 - i * 20
            color = (*self.COLOR_NUCLEUS[:3], alpha)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_rad, color)

        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_NUCLEUS)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_NUCLEUS)
        
        # Title
        if self.steps < 200:
            title_surf = self.font_title.render("Reach the Nucleus!", True, self.COLOR_TEXT)
            self.screen.blit(title_surf, (self.WIDTH/2 - title_surf.get_width()/2, pos[1] - 80))


    def _render_pseudopod(self):
        if self.pseudopod_length > 0:
            start_pos = (int(self.player_pos.x), int(self.player_pos.y - self.camera_y))
            end_pos_vec = self.player_pos + self.aim_direction * self.pseudopod_length
            end_pos = (int(end_pos_vec.x), int(end_pos_vec.y - self.camera_y))
            
            # Draw as a series of circles for a blobby look
            points = 10
            for i in range(points):
                t = i / (points - 1)
                interp_pos = start_pos[0] + (end_pos[0]-start_pos[0])*t, start_pos[1] + (end_pos[1]-start_pos[1])*t
                radius = int(max(1, (1 - t) * 6 + 2))
                pygame.gfxdraw.filled_circle(self.screen, int(interp_pos[0]), int(interp_pos[1]), radius, self.COLOR_PSEUDOPOD)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'].x), int(p['pos'].y - self.camera_y))
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'][:3], alpha)
            if p['radius'] > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), color)

    def _render_ui(self):
        # Score and Steps
        score_text = self.font_ui.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        steps_text = self.font_ui.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(steps_text, (10, 30))
        
        # DNA Progress Bar
        bar_x, bar_y, bar_w, bar_h = 10, 55, 200, 20
        pygame.draw.rect(self.screen, self.COLOR_BG_ACCENT, (bar_x, bar_y, bar_w, bar_h))
        fill_w = min(bar_w, (self.dna_collected / self.DNA_TO_UNLOCK_CILIA_BOOST) * bar_w)
        pygame.draw.rect(self.screen, self.COLOR_DNA, (bar_x, bar_y, fill_w, bar_h))
        dna_text = self.font_ability.render(f"DNA: {self.dna_collected}", True, self.COLOR_TEXT)
        self.screen.blit(dna_text, (bar_x + 5, bar_y + 3))

        # Ability Icons
        dj_color = self.COLOR_ABILITY_UNLOCKED if self.unlocked_abilities["double_jump"] else self.COLOR_ABILITY_LOCKED
        cb_color = self.COLOR_ABILITY_UNLOCKED if self.unlocked_abilities["cilia_boost"] else self.COLOR_ABILITY_LOCKED
        
        dj_text = self.font_ability.render(f"Double Jump", True, dj_color)
        cb_text = self.font_ability.render(f"Cilia Boost", True, cb_color)
        self.screen.blit(dj_text, (self.WIDTH - dj_text.get_width() - 10, 10))
        self.screen.blit(cb_text, (self.WIDTH - cb_text.get_width() - 10, 30))

        # Last reward info
        if self.last_reward_info:
            reward_text = self.font_ui.render(self.last_reward_info, True, self.COLOR_PLAYER)
            self.screen.blit(reward_text, (self.WIDTH/2 - reward_text.get_width()/2, 10))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "dna_collected": self.dna_collected,
            "player_pos": (self.player_pos.x, self.player_pos.y),
            "unlocked_double_jump": self.unlocked_abilities["double_jump"],
            "unlocked_cilia_boost": self.unlocked_abilities["cilia_boost"],
        }

    def _create_particles(self, position, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': position.copy(),
                'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'life': life,
                'max_life': life,
                'color': color,
                'radius': self.np_random.uniform(2, 5)
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # --- Example Usage ---
    env = GameEnv(render_mode="rgb_array")
    
    # To play manually, you need a different setup since this is an rgb_array env.
    # This example just runs a random agent and prints info.
    
    # For a proper interactive window:
    # Set the SDL_VIDEODRIVER to a real driver if you unset the dummy one at the top.
    # For example, on Linux: os.environ["SDL_VIDEODRIVER"] = "x11"
    # Or just remove the os.environ.setdefault line.
    
    # Re-initialize pygame with a display
    pygame.quit()
    pygame.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Cellular Evolution Environment")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    # Key mapping for human play
    key_map = {
        pygame.K_LEFT: 3, pygame.K_RIGHT: 4,
        pygame.K_UP: 1, pygame.K_DOWN: 2
    }
    
    running = True
    while running:
        movement_action = 0 # no-op
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        elif keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2

        if keys[pygame.K_SPACE]: space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_action = 1
            
        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}, Steps: {info['steps']}")
            obs, info = env.reset()

        clock.tick(30) # Run at 30 FPS

    env.close()