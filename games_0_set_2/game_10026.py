import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import numpy as np
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Navigate a cyberpunk world, using EMP grenades to disable laser grids and reach the exit."
    )
    user_guide = (
        "Use arrow keys to move. Hold Shift to aim and use arrow keys to adjust the angle. "
        "Press Space to launch an EMP grenade."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width = 640
        self.height = 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_obj = pygame.font.Font(None, 32)
        
        # Colors
        self.COLOR_BG = (15, 10, 30)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (0, 100, 200)
        self.COLOR_LASER = (255, 0, 50)
        self.COLOR_LASER_GLOW = (200, 0, 40)
        self.COLOR_TARGET = (50, 255, 50)
        self.COLOR_TARGET_GLOW = (40, 200, 40)
        self.COLOR_GRENADE = (100, 200, 255)
        self.COLOR_GRENADE_PULSE = (50, 150, 255)
        self.COLOR_NEON_1 = (255, 0, 255)
        self.COLOR_NEON_2 = (200, 50, 200)
        self.COLOR_UI = (220, 220, 255)
        self.COLOR_DISABLED_LASER = (50, 50, 60)

        # Game state that persists across resets
        self.current_level_num = 1
        self.unlocked_gadgets = {
            'faster_cooldown': False,
            'larger_pulse': False,
        }
        
        self._define_levels()

        # Initialize state variables
        # These will be properly set in reset()
        self.player_pos = None
        self.player_vel = None
        self.aim_angle = None
        self.grenades = []
        self.particles = []
        self.lasers = []
        self.buildings = []
        self.target_pos = None
        self.target_radius = None
        self.grenade_cooldown = 0
        self.base_grenade_cooldown = 90 # 3 seconds at 30fps
        self.grenade_pulse_radius = 60
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.is_aiming = False

    def _define_levels(self):
        self.levels = [
            { # Level 1
                "player_start": [80, 200], "target_pos": [560, 200],
                "lasers": [
                    {'p1': (320, 50), 'p2': (320, 350), 'offset': 0},
                ],
                "buildings": [pygame.Rect(200, 150, 20, 100), pygame.Rect(420, 150, 20, 100)]
            },
            { # Level 2
                "player_start": [80, 80], "target_pos": [560, 320],
                "lasers": [
                    {'p1': (50, 200), 'p2': (300, 200), 'offset': 0},
                    {'p1': (340, 200), 'p2': (590, 200), 'offset': 90},
                ],
                "buildings": [pygame.Rect(310, 100, 20, 200)]
            },
            { # Level 3
                "player_start": [80, 200], "target_pos": [560, 200],
                "lasers": [
                    {'p1': (220, 50), 'p2': (220, 350), 'offset': 0},
                    {'p1': (420, 50), 'p2': (420, 350), 'offset': 60},
                    {'p1': (220, 195), 'p2': (420, 195), 'offset': 30},
                ],
                "buildings": []
            }
        ]

    def _load_level(self):
        level_data = self.levels[(self.current_level_num - 1) % len(self.levels)]
        
        self.player_pos = list(level_data["player_start"])
        self.target_pos = list(level_data["target_pos"])
        self.target_radius = 25
        
        self.buildings = [
            pygame.Rect(0, 0, self.width, 10), pygame.Rect(0, self.height - 10, self.width, 10),
            pygame.Rect(0, 0, 10, self.height), pygame.Rect(self.width - 10, 0, 10, self.height)
        ] + level_data["buildings"]

        # Difficulty scaling
        level_tier = (self.current_level_num - 1) // 3
        on_time = max(90 - level_tier * 10, 30) # 3s -> 1s min
        off_time = max(60 - level_tier * 10, 30) # 2s -> 1s min

        self.lasers = []
        for laser_def in level_data["lasers"]:
            self.lasers.append({
                'p1': laser_def['p1'], 'p2': laser_def['p2'],
                'timer': laser_def['offset'], 'cycle_on': on_time, 'cycle_off': off_time,
                'is_on': False, 'disabled_timer': 0
            })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_vel = [0, 0]
        self.aim_angle = 0
        self.grenades = []
        self.particles = []
        self.grenade_cooldown = 0
        
        # Apply gadget effects
        if self.unlocked_gadgets['faster_cooldown']:
            self.base_grenade_cooldown = 60 # 2 seconds
        if self.unlocked_gadgets['larger_pulse']:
            self.grenade_pulse_radius = 80

        self._load_level()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.01 # Survival reward
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.is_aiming = shift_held # Store for rendering
        
        # --- Handle Input & Player Logic ---
        self._handle_input(movement, space_held, shift_held)
        self._update_player()
        
        # --- Handle Game Object Logic ---
        self._update_grenades()
        self._update_lasers()
        self._update_particles()
        
        # --- Handle Interactions & Rewards ---
        reward += self._check_grenade_effects()
        if self.grenade_cooldown > 0:
            self.grenade_cooldown -= 1
            reward -= 0.1
        
        # --- Check Termination Conditions ---
        terminated = False
        if self._check_player_laser_collision():
            # SFX: Player death sizzle
            reward = -100
            terminated = True
        elif self._check_player_target_collision():
            # SFX: Level complete fanfare
            reward = 100
            terminated = True
            self.current_level_num += 1
            self._check_gadget_unlocks()

        self.steps += 1
        if self.steps >= 1000:
            terminated = True
        
        self.game_over = terminated
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
        
    def _handle_input(self, movement, space_held, shift_held):
        player_speed = 2.0
        if shift_held:
            # Aiming mode
            aim_speed = 0.1
            if movement == 1: self.aim_angle -= aim_speed # Up -> Counter-clockwise
            elif movement == 2: self.aim_angle += aim_speed # Down -> Clockwise
            elif movement == 3: self.aim_angle -= aim_speed # Left -> Counter-clockwise
            elif movement == 4: self.aim_angle += aim_speed # Right -> Clockwise
        else:
            # Movement mode
            if movement == 1: self.player_vel[1] -= player_speed
            elif movement == 2: self.player_vel[1] += player_speed
            elif movement == 3: self.player_vel[0] -= player_speed
            elif movement == 4: self.player_vel[0] += player_speed
            
        if space_held and self.grenade_cooldown == 0:
            self._launch_grenade()

    def _update_player(self):
        self.player_pos[0] += self.player_vel[0]
        self.player_pos[1] += self.player_vel[1]
        self.player_vel[0] *= 0.85 # Friction
        self.player_vel[1] *= 0.85
        
        # Boundary collision
        self.player_pos[0] = np.clip(self.player_pos[0], 10, self.width - 10)
        self.player_pos[1] = np.clip(self.player_pos[1], 10, self.height - 10)

    def _launch_grenade(self):
        # SFX: Grenade launch
        self.grenade_cooldown = self.base_grenade_cooldown
        vel = [math.cos(self.aim_angle) * 8, math.sin(self.aim_angle) * 8]
        self.grenades.append({
            'pos': list(self.player_pos),
            'vel': vel,
            'lifetime': 60, # 2 seconds
            'state': 'flying' # 'flying', 'pulsing'
        })

    def _update_grenades(self):
        for g in self.grenades[:]:
            if g['state'] == 'flying':
                g['pos'][0] += g['vel'][0]
                g['pos'][1] += g['vel'][1]
                g['lifetime'] -= 1
                
                # Wall collision
                if not (10 < g['pos'][0] < self.width - 10 and 10 < g['pos'][1] < self.height - 10):
                    g['state'] = 'pulsing'
                    g['lifetime'] = 20 # Pulse duration
                    self._create_particles(g['pos'], 30)
                    # SFX: Grenade impact
                elif g['lifetime'] <= 0:
                    g['state'] = 'pulsing'
                    g['lifetime'] = 20
                    self._create_particles(g['pos'], 30)
                    # SFX: Grenade impact
            
            elif g['state'] == 'pulsing':
                g['lifetime'] -= 1
                if g['lifetime'] <= 0:
                    self.grenades.remove(g)

    def _update_lasers(self):
        for laser in self.lasers:
            if laser['disabled_timer'] > 0:
                laser['disabled_timer'] -= 1
                laser['is_on'] = False
                continue
            
            laser['timer'] += 1
            total_cycle = laser['cycle_on'] + laser['cycle_off']
            if laser['timer'] % total_cycle < laser['cycle_on']:
                if not laser['is_on']: # SFX: Laser turns on
                    pass
                laser['is_on'] = True
            else:
                if laser['is_on']: # SFX: Laser turns off
                    pass
                laser['is_on'] = False

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)

    def _check_grenade_effects(self):
        reward = 0
        for g in self.grenades:
            if g['state'] == 'pulsing':
                pulse_radius = self.grenade_pulse_radius * (1 - (g['lifetime'] / 20))
                for laser in self.lasers:
                    if laser['disabled_timer'] == 0:
                        # Simple line-circle intersection
                        p1 = np.array(laser['p1'])
                        p2 = np.array(laser['p2'])
                        g_pos = np.array(g['pos'])
                        
                        d = np.linalg.norm(np.cross(p2-p1, p1-g_pos))/np.linalg.norm(p2-p1) if np.linalg.norm(p2-p1) > 0 else np.linalg.norm(p1-g_pos)

                        if d < pulse_radius:
                            laser['disabled_timer'] = 150 # 5 seconds
                            reward += 5
                            # SFX: Laser disable sound
        return reward

    def _check_player_laser_collision(self):
        player_radius = 10
        for laser in self.lasers:
            if laser['is_on']:
                p1 = np.array(laser['p1'])
                p2 = np.array(laser['p2'])
                p_pos = np.array(self.player_pos)
                
                # Check distance from line segment
                l2 = np.sum((p1-p2)**2)
                if l2 == 0.0:
                    d = np.linalg.norm(p_pos - p1)
                else:
                    t = max(0, min(1, np.dot(p_pos - p1, p2 - p1) / l2))
                    projection = p1 + t * (p2 - p1)
                    d = np.linalg.norm(p_pos - projection)
                
                if d < player_radius:
                    return True
        return False

    def _check_player_target_collision(self):
        dist = math.hypot(self.player_pos[0] - self.target_pos[0], self.player_pos[1] - self.target_pos[1])
        return dist < (10 + self.target_radius)

    def _check_gadget_unlocks(self):
        if self.current_level_num == 4 and not self.unlocked_gadgets['faster_cooldown']:
            self.unlocked_gadgets['faster_cooldown'] = True
        if self.current_level_num == 7 and not self.unlocked_gadgets['larger_pulse']:
            self.unlocked_gadgets['larger_pulse'] = True

    def _create_particles(self, pos, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(20, 41)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifetime': lifetime})
            
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Buildings
        for building in self.buildings:
            pygame.draw.rect(self.screen, self.COLOR_NEON_2, building)
            pygame.draw.rect(self.screen, self.COLOR_NEON_1, building, 2)
        
        # Target
        self._draw_glow_circle(self.screen, self.COLOR_TARGET_GLOW, self.COLOR_TARGET, self.target_pos, self.target_radius)

        # Lasers
        for laser in self.lasers:
            if laser['disabled_timer'] > 0:
                pygame.gfxdraw.aaline(self.screen, int(laser['p1'][0]), int(laser['p1'][1]), int(laser['p2'][0]), int(laser['p2'][1]), self.COLOR_DISABLED_LASER)
            elif laser['is_on']:
                self._draw_glow_line(self.screen, self.COLOR_LASER_GLOW, self.COLOR_LASER, laser['p1'], laser['p2'], 4)
        
        # Grenades
        for g in self.grenades:
            if g['state'] == 'flying':
                self._draw_glow_circle(self.screen, self.COLOR_GRENADE, self.COLOR_GRENADE, g['pos'], 5)
            elif g['state'] == 'pulsing':
                pulse_progress = 1 - (g['lifetime'] / 20)
                radius = self.grenade_pulse_radius * pulse_progress
                alpha = 150 * (1 - pulse_progress)
                color = (*self.COLOR_GRENADE_PULSE, int(alpha))
                
                surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (radius, radius), int(radius))
                self.screen.blit(surf, (g['pos'][0] - radius, g['pos'][1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Particles
        for p in self.particles:
            alpha = max(0, 255 * (p['lifetime'] / 40.0))
            color = (*self.COLOR_GRENADE, int(alpha))
            size = max(0, 3 * (p['lifetime'] / 40.0))
            rect = pygame.Rect(p['pos'][0] - size/2, p['pos'][1] - size/2, size, size)
            pygame.draw.rect(self.screen, color, rect)

        # Player
        self._draw_glow_circle(self.screen, self.COLOR_PLAYER_GLOW, self.COLOR_PLAYER, self.player_pos, 10)

        # Aiming reticle
        if self.is_aiming:
             end_pos = (self.player_pos[0] + math.cos(self.aim_angle) * 50,
                        self.player_pos[1] + math.sin(self.aim_angle) * 50)
             pygame.draw.aaline(self.screen, self.COLOR_UI, self.player_pos, end_pos)

    def _render_ui(self):
        # Objective
        obj_text = self.font_obj.render("Reach the Green Target", True, self.COLOR_UI)
        self.screen.blit(obj_text, (20, 20))

        # Score
        score_text = self.font_ui.render(f"Score: {self.score:.0f}", True, self.COLOR_UI)
        score_rect = score_text.get_rect(topright=(self.width - 20, 20))
        self.screen.blit(score_text, score_rect)

        # Level
        level_text = self.font_ui.render(f"Level: {self.current_level_num}", True, self.COLOR_UI)
        level_rect = level_text.get_rect(topright=(self.width - 20, 45))
        self.screen.blit(level_text, level_rect)
        
        # Cooldown
        if self.grenade_cooldown > 0:
            cooldown_text = self.font_ui.render(f"Grenade CD: {self.grenade_cooldown/30:.1f}s", True, self.COLOR_UI)
            cd_rect = cooldown_text.get_rect(bottomleft=(20, self.height - 20))
            self.screen.blit(cooldown_text, cd_rect)

    def _draw_glow_circle(self, surface, glow_color, main_color, center, radius):
        center_i = (int(center[0]), int(center[1]))
        # Glow
        glow_radius = int(radius * 1.8)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*glow_color, 80), (glow_radius, glow_radius), glow_radius)
        surface.blit(s, (center_i[0] - glow_radius, center_i[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        # Main circle
        pygame.gfxdraw.filled_circle(surface, center_i[0], center_i[1], int(radius), main_color)
        pygame.gfxdraw.aacircle(surface, center_i[0], center_i[1], int(radius), main_color)
    
    def _draw_glow_line(self, surface, glow_color, main_color, p1, p2, width):
        p1_i = (int(p1[0]), int(p1[1]))
        p2_i = (int(p2[0]), int(p2[1]))
        # Glow line (thicker, lower alpha)
        pygame.draw.line(surface, (*glow_color, 100), p1_i, p2_i, width + 4)
        # Main line
        pygame.draw.line(surface, main_color, p1_i, p2_i, width)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.current_level_num,
            "grenade_cooldown": self.grenade_cooldown,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment.
    os.environ.pop("SDL_VIDEODRIVER", None)
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Cyberpunk Stealth")
    
    terminated = False
    clock = pygame.time.Clock()
    
    while not terminated:
        movement = 0 # No-op
        space_action = 0
        shift_action = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                # Use a flag to break out of the outer loop as well
                should_quit = True

        if terminated:
            break

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]:
            space_action = 1
            
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1
            
        action = [movement, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # The observation is the rendered frame, so we just need to display it
        # Pygame uses (width, height), but our obs is (height, width, 3).
        # We need to transpose it back for display.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Level: {info['level']}")
            pygame.time.wait(2000)
            # Check for quit event during wait
            quit_during_wait = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_during_wait = True
            if quit_during_wait:
                break
            obs, info = env.reset()
            terminated = False

        clock.tick(30) # Run at 30 FPS

    env.close()