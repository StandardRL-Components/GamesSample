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
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Navigate a dark facility, avoiding sonar-equipped guardians. "
        "Use teleportation and sound-dampening abilities to reach the exit undetected."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to teleport in your current direction. "
        "Press shift to dampen the sound of your next step."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000

    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_WALL = (40, 50, 70)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255, 50)
    COLOR_GUARDIAN = (255, 50, 50)
    COLOR_GUARDIAN_GLOW = (255, 50, 50, 70)
    COLOR_SONAR = (255, 80, 80, 40)
    COLOR_EXIT = (180, 0, 255)
    COLOR_EXIT_GLOW = (180, 0, 255, 60)
    COLOR_SOUND = (255, 200, 0)
    COLOR_CHECKPOINT = (0, 255, 100)
    COLOR_CHECKPOINT_GLOW = (0, 255, 100, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_PARTICLE = (200, 200, 255)

    # Game Parameters
    PLAYER_SPEED = 4.0
    PLAYER_RADIUS = 10
    GUARDIAN_RADIUS = 12
    GUARDIAN_BASE_SPEED = 0.5
    SONAR_RADIUS = 150
    SONAR_ANGLE_DEG = 60
    INITIAL_TELEPORT_CHARGES = 3
    TELEPORT_DISTANCE = 100 # Approx 5 units of 20px
    ABILITY_DAMPEN_COOLDOWN = 150 # 5 seconds at 30fps
    ABILITY_DAMPEN_FACTOR = 0.2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        self.render_mode = render_mode

        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.array([0.0, 0.0])
        self.player_target_pos = np.array([0.0, 0.0])
        self.player_last_move_dir = np.array([1.0, 0.0])
        self.teleport_charges = 0
        self.guardians = []
        self.walls = []
        self.exit_pos = np.array([0.0, 0.0])
        self.checkpoints = []
        self.sound_waves = []
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.ability_dampen_active = False
        self.ability_dampen_cooldown = 0
        self.unlocked_abilities = set()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self._setup_level()

        self.player_target_pos = self.player_pos.copy()
        self.player_last_move_dir = np.array([1.0, 0.0])
        self.teleport_charges = self.INITIAL_TELEPORT_CHARGES
        
        self.sound_waves.clear()
        self.particles.clear()

        self.prev_space_held = False
        self.prev_shift_held = False
        self.ability_dampen_active = False
        self.ability_dampen_cooldown = 0
        self.unlocked_abilities = {'dampen'} # Start with dampen unlocked

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            obs = self._get_observation()
            return obs, 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.01 # Small time penalty
        
        self._handle_input(movement, space_held, shift_held)
        self._update_player()
        self._update_guardians()
        self._update_sound_waves()
        self._update_particles()
        self._update_abilities()

        # Check for events and update rewards
        reward += self._check_checkpoint_collisions()
        
        terminated = self._check_termination()
        if terminated:
            if self._is_player_at_exit():
                reward += 100.0 # Win
                self._create_particles(self.player_pos, 100, self.COLOR_EXIT, 2.0)
            elif self._is_player_detected():
                reward -= 100.0 # Loss
                self._create_particles(self.player_pos, 100, self.COLOR_GUARDIAN, 2.0)
        
        self.score += reward
        self.steps += 1
        self.game_over = terminated

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    # --- Update Logic ---

    def _handle_input(self, movement, space_held, shift_held):
        # Movement
        move_dir = np.array([0.0, 0.0])
        if movement == 1: move_dir[1] = -1.0  # Up
        elif movement == 2: move_dir[1] = 1.0  # Down
        elif movement == 3: move_dir[0] = -1.0  # Left
        elif movement == 4: move_dir[0] = 1.0  # Right
        
        if np.any(move_dir):
            self.player_target_pos = self.player_pos + move_dir * self.PLAYER_SPEED * 5 # Target 5 steps ahead
            self.player_last_move_dir = move_dir

        # Teleport (on press)
        if space_held and not self.prev_space_held and self.teleport_charges > 0:
            self._teleport_player()

        # Ability (on press)
        if shift_held and not self.prev_shift_held and self.ability_dampen_cooldown == 0:
            self.ability_dampen_active = True
            # sfx: Ability activate sound

    def _update_player(self):
        # Smooth movement towards target
        move_vec = self.player_target_pos - self.player_pos
        dist = np.linalg.norm(move_vec)
        
        if dist > 1.0:
            move_step = move_vec / dist * min(dist, self.PLAYER_SPEED)
            next_pos = self.player_pos + move_step
            
            # Wall collision
            player_rect = pygame.Rect(0, 0, self.PLAYER_RADIUS * 2, self.PLAYER_RADIUS * 2)
            player_rect.center = (int(next_pos[0]), int(next_pos[1]))
            
            collided_wall = False
            for wall in self.walls:
                if wall.colliderect(player_rect):
                    collided_wall = True
                    break
            
            if not collided_wall:
                self.player_pos = next_pos
                # Create sound wave on movement
                if self.steps % 4 == 0: # Emit sound periodically while moving
                    intensity = self.ABILITY_DAMPEN_FACTOR if self.ability_dampen_active else 1.0
                    self._create_sound_wave(self.player_pos, 80 * intensity, 1.0 * intensity)
                    if self.ability_dampen_active:
                        self.ability_dampen_active = False # One-shot use
                        self.ability_dampen_cooldown = self.ABILITY_DAMPEN_COOLDOWN
                        # sfx: Dampened step sound
                    else:
                        pass # sfx: Normal step sound
            else:
                self.player_target_pos = self.player_pos.copy() # Stop if hitting a wall
        else:
             self.player_target_pos = self.player_pos.copy()

        # Keep player within bounds
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.SCREEN_WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS)
        # self.player_target_pos = self.player_pos.copy() # This was causing stuttering

    def _teleport_player(self):
        self.teleport_charges -= 1
        # sfx: Teleport whoosh
        self._create_particles(self.player_pos, 50, self.COLOR_PLAYER, 1.5)
        
        target_pos = self.player_pos + self.player_last_move_dir * self.TELEPORT_DISTANCE
        
        # Check for wall collisions along the teleport path
        for i in range(1, 11):
            check_pos = self.player_pos + self.player_last_move_dir * (self.TELEPORT_DISTANCE * i / 10)
            check_rect = pygame.Rect(0,0,self.PLAYER_RADIUS*2, self.PLAYER_RADIUS*2)
            check_rect.center = (int(check_pos[0]), int(check_pos[1]))
            
            collided = False
            for wall in self.walls:
                if wall.colliderect(check_rect):
                    collided = True
                    break

            if collided:
                target_pos = check_pos - self.player_last_move_dir * (self.PLAYER_RADIUS + 2)
                break

        self.player_pos = target_pos
        self.player_target_pos = self.player_pos.copy()
        
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.SCREEN_WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS)
        
        self._create_particles(self.player_pos, 50, self.COLOR_PLAYER, 1.5)
        self._create_sound_wave(self.player_pos, 150, 1.5)

    def _update_guardians(self):
        speed_multiplier = 1.0 + (self.steps // 1000) * 0.05
        
        for g in self.guardians:
            target_node = g['path'][g['path_idx']]
            target_pos = np.array(target_node, dtype=float)
            
            move_vec = target_pos - g['pos']
            dist = np.linalg.norm(move_vec)
            
            if dist < g['speed'] * speed_multiplier:
                g['path_idx'] = (g['path_idx'] + 1) % len(g['path'])
            else:
                g['pos'] += move_vec / dist * g['speed'] * speed_multiplier
            
            g['sonar_angle'] = (g['sonar_angle'] + g['sonar_rot_speed']) % 360

    def _update_sound_waves(self):
        for wave in self.sound_waves:
            wave['radius'] += wave['speed']
            wave['alpha'] = max(0, 255 * (1 - (wave['radius'] / wave['max_radius'])))
        self.sound_waves = [w for w in self.sound_waves if w['alpha'] > 0]

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def _update_abilities(self):
        if self.ability_dampen_cooldown > 0:
            self.ability_dampen_cooldown -= 1

    # --- State Checks & Events ---
    
    def _check_termination(self):
        return (
            self._is_player_at_exit() or
            self._is_player_detected()
        )

    def _is_player_at_exit(self):
        return np.linalg.norm(self.player_pos - self.exit_pos) < self.PLAYER_RADIUS + 15

    def _is_player_detected(self):
        player_rect = pygame.Rect(0, 0, self.PLAYER_RADIUS, self.PLAYER_RADIUS)
        player_rect.center = self.player_pos
        
        for g in self.guardians:
            # Check sonar detection
            dist_to_player = np.linalg.norm(self.player_pos - g['pos'])
            if dist_to_player < self.SONAR_RADIUS:
                angle_to_player_rad = math.atan2(self.player_pos[1] - g['pos'][1], self.player_pos[0] - g['pos'][0])
                angle_to_player_deg = (math.degrees(angle_to_player_rad) + 360) % 360
                
                sonar_start_angle = (g['sonar_angle'] - self.SONAR_ANGLE_DEG / 2 + 360) % 360
                sonar_end_angle = (g['sonar_angle'] + self.SONAR_ANGLE_DEG / 2 + 360) % 360
                
                # Handle angle wrapping
                in_sonar = False
                if sonar_start_angle < sonar_end_angle:
                    if sonar_start_angle <= angle_to_player_deg <= sonar_end_angle:
                        in_sonar = True
                else: # Wraps around 360
                    if angle_to_player_deg >= sonar_start_angle or angle_to_player_deg <= sonar_end_angle:
                        in_sonar = True
                
                if in_sonar:
                    # Line of sight check
                    if not self._is_line_of_sight_blocked(g['pos'], self.player_pos):
                        return True
            
            # Check sound detection
            for wave in self.sound_waves:
                dist_to_wave = np.linalg.norm(g['pos'] - wave['pos'])
                if dist_to_wave < wave['radius']:
                    if not self._is_line_of_sight_blocked(g['pos'], wave['pos']):
                        # sfx: Guardian alert sound
                        # Make guardian "look" towards the sound
                        angle_to_sound_deg = math.degrees(math.atan2(wave['pos'][1] - g['pos'][1], wave['pos'][0] - g['pos'][0]))
                        g['sonar_angle'] = angle_to_sound_deg
                        # For simplicity, sound only alerts, not a game over condition.
                        
        return False

    def _is_line_of_sight_blocked(self, p1, p2):
        for wall in self.walls:
            if wall.clipline(p1, p2):
                return True
        return False

    def _check_checkpoint_collisions(self):
        reward = 0.0
        player_rect = pygame.Rect(0, 0, self.PLAYER_RADIUS, self.PLAYER_RADIUS)
        player_rect.center = self.player_pos
        
        active_checkpoints = []
        for cp in self.checkpoints:
            if player_rect.colliderect(cp['rect']):
                if cp['ability'] not in self.unlocked_abilities:
                    self.unlocked_abilities.add(cp['ability'])
                    reward += 5.0
                    # sfx: Ability unlocked
                    self._create_particles(cp['rect'].center, 50, self.COLOR_CHECKPOINT)
                # For this demo, we'll just keep the checkpoint, but in a real game it would be consumed.
                else:
                    active_checkpoints.append(cp)
            else:
                active_checkpoints.append(cp)
        # self.checkpoints = active_checkpoints # Uncomment to make checkpoints one-time use
        return reward

    # --- Rendering ---
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _render_game(self):
        self._render_walls()
        self._render_exit()
        self._render_checkpoints()
        self._render_sound_waves()
        self._render_guardians()
        self._render_player()
        self._render_particles()

    def _render_walls(self):
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)

    def _render_exit(self):
        pulse = (math.sin(self.steps * 0.1) + 1) / 2 # 0 to 1
        radius = 15 + pulse * 5
        glow_radius = int(radius * 2.5)
        self._render_glow_circle(self.screen, self.COLOR_EXIT_GLOW, self.exit_pos, glow_radius)
        pygame.gfxdraw.filled_circle(self.screen, int(self.exit_pos[0]), int(self.exit_pos[1]), int(radius), self.COLOR_EXIT)
        pygame.gfxdraw.aacircle(self.screen, int(self.exit_pos[0]), int(self.exit_pos[1]), int(radius), self.COLOR_EXIT)

    def _render_checkpoints(self):
        for cp in self.checkpoints:
            if cp['ability'] not in self.unlocked_abilities:
                pulse = (math.sin(self.steps * 0.15) + 1) / 2
                color = self.COLOR_CHECKPOINT
                glow_color = self.COLOR_CHECKPOINT_GLOW
                
                pygame.draw.rect(self.screen, color, cp['rect'], border_radius=4)
                
                # Simple glow effect by drawing a larger, transparent rect
                glow_rect = cp['rect'].inflate(10 + pulse * 5, 10 + pulse * 5)
                s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
                pygame.draw.rect(s, glow_color, s.get_rect(), border_radius=8)
                self.screen.blit(s, glow_rect.topleft)

    def _render_guardians(self):
        for g in self.guardians:
            # Body
            pos_int = (int(g['pos'][0]), int(g['pos'][1]))
            self._render_glow_circle(self.screen, self.COLOR_GUARDIAN_GLOW, pos_int, self.GUARDIAN_RADIUS * 2)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.GUARDIAN_RADIUS, self.COLOR_GUARDIAN)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.GUARDIAN_RADIUS, self.COLOR_GUARDIAN)
            
            # Sonar Cone
            angle_rad = math.radians(g['sonar_angle'])
            half_arc_rad = math.radians(self.SONAR_ANGLE_DEG / 2)
            p1 = pos_int
            p2 = (pos_int[0] + self.SONAR_RADIUS * math.cos(angle_rad - half_arc_rad),
                  pos_int[1] + self.SONAR_RADIUS * math.sin(angle_rad - half_arc_rad))
            p3 = (pos_int[0] + self.SONAR_RADIUS * math.cos(angle_rad + half_arc_rad),
                  pos_int[1] + self.SONAR_RADIUS * math.sin(angle_rad + half_arc_rad))
            
            pygame.gfxdraw.filled_trigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_SONAR)

    def _render_player(self):
        pos_int = (int(self.player_pos[0]), int(self.player_pos[1]))
        
        # Glow
        glow_radius = int(self.PLAYER_RADIUS * 2.5)
        self._render_glow_circle(self.screen, self.COLOR_PLAYER_GLOW, pos_int, glow_radius)
        
        # Body
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        
        # Direction indicator
        dir_end = self.player_pos + self.player_last_move_dir * self.PLAYER_RADIUS
        pygame.draw.line(self.screen, self.COLOR_BG, pos_int, (int(dir_end[0]), int(dir_end[1])), 2)

    def _render_sound_waves(self):
        for wave in self.sound_waves:
            color = (*self.COLOR_SOUND, int(wave['alpha']))
            pos = (int(wave['pos'][0]), int(wave['pos'][1]))
            radius = int(wave['radius'])
            if radius > 1:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
                if radius > 2:
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius - 2, (*color[:3], color[3] // 2))

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            color = (*p['color'], int(alpha))
            s = pygame.Surface((3, 3), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (1, 1), 1)
            self.screen.blit(s, (int(p['pos'][0]), int(p['pos'][1])), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score:.0f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Steps
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))

        # Teleport Charges
        teleport_text = self.font_small.render("TELEPORT", True, self.COLOR_TEXT)
        self.screen.blit(teleport_text, (10, 40))
        for i in range(self.INITIAL_TELEPORT_CHARGES):
            color = self.COLOR_PLAYER if i < self.teleport_charges else self.COLOR_WALL
            pygame.draw.rect(self.screen, color, (10 + i * 20, 60, 15, 15), border_radius=3)
            
        # Ability Status
        ability_text = self.font_small.render("DAMPEN", True, self.COLOR_TEXT)
        self.screen.blit(ability_text, (10, 85))
        if self.ability_dampen_cooldown > 0:
            cooldown_ratio = self.ability_dampen_cooldown / self.ABILITY_DAMPEN_COOLDOWN
            pygame.draw.rect(self.screen, self.COLOR_WALL, (10, 105, 70, 15))
            pygame.draw.rect(self.screen, self.COLOR_GUARDIAN, (10, 105, 70 * cooldown_ratio, 15))
        elif self.ability_dampen_active:
            pygame.draw.rect(self.screen, self.COLOR_CHECKPOINT, (10, 105, 70, 15))
        else:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, (10, 105, 70, 15))

    # --- Helpers & Setup ---

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "teleport_charges": self.teleport_charges,
            "player_pos": self.player_pos,
        }

    def _setup_level(self):
        self.player_pos = np.array([50.0, self.SCREEN_HEIGHT / 2.0])
        self.exit_pos = np.array([self.SCREEN_WIDTH - 50.0, self.SCREEN_HEIGHT / 2.0])

        self.walls = [
            pygame.Rect(150, 0, 30, 150),
            pygame.Rect(150, self.SCREEN_HEIGHT - 150, 30, 150),
            pygame.Rect(self.SCREEN_WIDTH - 180, 0, 30, 150),
            pygame.Rect(self.SCREEN_WIDTH - 180, self.SCREEN_HEIGHT - 150, 30, 150),
        ]
        
        self.guardians = [
            {
                'pos': np.array([self.SCREEN_WIDTH / 2.0, 100.0]),
                'path': [(self.SCREEN_WIDTH / 2, 100), (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 100)],
                'path_idx': 0,
                'speed': self.GUARDIAN_BASE_SPEED,
                'sonar_angle': 0,
                'sonar_rot_speed': 0.75,
            }
        ]
        
        # A dummy checkpoint for reward testing. In a full game, this would unlock a new ability.
        self.checkpoints = [
            {'rect': pygame.Rect(self.SCREEN_WIDTH - 100, self.SCREEN_HEIGHT - 60, 40, 40), 'ability': 'teleport_recharge'}
        ]

    def _create_sound_wave(self, pos, max_radius, intensity):
        self.sound_waves.append({
            'pos': pos.copy(),
            'radius': 0.0,
            'max_radius': max_radius,
            'speed': 1.5 * intensity,
            'alpha': 255,
        })

    def _create_particles(self, pos, count, color, speed_mult=1.0):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2.0) * speed_mult
            life = random.randint(20, 40)
            self.particles.append({
                'pos': pos.copy(),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'life': life,
                'max_life': life,
                'color': color,
            })

    def _render_glow_circle(self, surface, color, center, radius, width=0):
        s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, color, (radius, radius), radius, width)
        surface.blit(s, (center[0] - radius, center[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def close(self):
        pygame.quit()
        
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # This part is for human playback and will not run in the test environment
    if os.environ.get("SDL_VIDEODRIVER") != "dummy":
        pygame.display.init()
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Sonar Stealth")
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0
        
        while running:
            # Map keyboard to MultiDiscrete action space
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated or truncated:
                print(f"Episode finished. Total reward: {total_reward:.2f}, Steps: {info['steps']}")
                total_reward = 0
                obs, info = env.reset()
                pygame.time.wait(2000) # Pause before restarting
                
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            clock.tick(env.metadata["render_fps"])
            
        env.close()