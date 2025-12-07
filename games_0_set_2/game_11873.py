import gymnasium as gym
import os
import pygame
import math
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "A rhythmic stealth game where you teleport between platforms to the beat. "
        "Avoid enemies and disappearing platforms to reach the exit."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to select a target platform. "
        "Press space to teleport on the beat."
    )
    auto_advance = True

    # --- RENDER CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    COLOR_BG = (15, 18, 32)
    COLOR_GRID = (30, 35, 58)
    COLOR_PLAYER = (0, 192, 255)
    COLOR_PLAYER_GLOW = (0, 192, 255, 50)
    COLOR_ENEMY = (255, 100, 0)
    COLOR_EXIT = (160, 32, 240)
    COLOR_PLATFORM_INACTIVE = (40, 50, 70)
    COLOR_PLATFORM_ACTIVE = (0, 255, 127)
    COLOR_PLATFORM_CHAIN = (255, 215, 0)
    COLOR_MARKER = (255, 255, 255)
    COLOR_TEXT = (220, 220, 240)

    # --- GAMEPLAY CONSTANTS ---
    BEAT_DURATION = 30  # 1 beat per second at 30 FPS
    MAX_STEPS = 1000
    PLAYER_RADIUS = 12
    ENEMY_RADIUS = 8
    PLATFORM_RADIUS = 20
    TELEPORT_DIST_MAX = 200
    PARTICLE_LIFESPAN = 20

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_big = pygame.font.SysFont("Consolas", 48, bold=True)

        self.level = 1
        self.unlocked_visuals = 0
        self.persistent_score = 0
        
        # State variables initialized in reset
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.beat_timer = 0
        self.platforms = []
        self.enemies = []
        self.particles = []
        self.player_platform_idx = 0
        self.player_target_platform_idx = 0
        self.player_pos = (0, 0)
        self.player_on_exit = False
        self.teleport_queued = False
        self.prev_space_held = False
        self.is_teleporting = False
        self.teleport_anim_timer = 0
        self.teleport_start_pos = (0, 0)
        self.teleport_end_pos = (0, 0)
        self.start_platform_idx = 0
        self.exit_platform_idx = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_on_exit = False
        self.beat_timer = 0
        
        self._generate_level()

        start_platform = self.platforms[self.start_platform_idx]
        self.player_pos = start_platform.pos
        self.player_platform_idx = self.start_platform_idx
        self.player_target_platform_idx = self.start_platform_idx

        self.teleport_queued = False
        self.prev_space_held = False
        self.is_teleporting = False
        self.teleport_anim_timer = 0
        self.particles.clear()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If already terminated, just return the final state
            terminated = True
            truncated = self.steps >= self.MAX_STEPS
            return self._get_observation(), 0, terminated, truncated, self._get_info()

        reward = 0
        self.steps += 1

        self._handle_input(action)
        self._update_animations()
        self._update_particles()
        
        self.beat_timer = (self.beat_timer + 1)
        if self.beat_timer >= self.BEAT_DURATION:
            self.beat_timer = 0
            reward += self._on_beat()

        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated:
            if self.player_on_exit:
                reward += 100
                self.score += 100
                self.persistent_score += 100
            elif self.game_over: # Died
                reward -= 100
                self.score -= 100

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_held, _ = action
        space_held = space_held == 1

        if movement != 0 and not self.is_teleporting:
            self._move_marker(movement)

        space_pressed = space_held and not self.prev_space_held
        if space_pressed and not self.is_teleporting and self.player_target_platform_idx != self.player_platform_idx:
            self.teleport_queued = True
        
        self.prev_space_held = space_held

    def _move_marker(self, direction):
        current_pos = self.platforms[self.player_platform_idx].pos
        best_target = -1
        min_score = float('inf')

        for i, p in enumerate(self.platforms):
            if i == self.player_platform_idx:
                continue

            dx, dy = p.pos[0] - current_pos[0], p.pos[1] - current_pos[1]
            dist = math.hypot(dx, dy)
            if dist == 0 or dist > self.TELEPORT_DIST_MAX:
                continue

            angle = math.atan2(-dy, dx) # Pygame y-axis is inverted
            
            # 1=up, 2=down, 3=left, 4=right
            if direction == 1 and abs(angle - math.pi/2) < math.pi/4: # UP
                score = dist * (1 + abs(angle - math.pi/2))
            elif direction == 2 and abs(angle + math.pi/2) < math.pi/4: # DOWN
                score = dist * (1 + abs(angle + math.pi/2))
            elif direction == 3 and (abs(angle - math.pi) < math.pi/4 or abs(angle + math.pi) < math.pi/4): # LEFT
                score = dist * (1 + min(abs(angle-math.pi), abs(angle+math.pi)))
            elif direction == 4 and abs(angle) < math.pi/4: # RIGHT
                score = dist * (1 + abs(angle))
            else:
                continue

            if score < min_score:
                min_score = score
                best_target = i
        
        if best_target != -1:
            self.player_target_platform_idx = best_target

    def _on_beat(self):
        beat_reward = 0
        
        # 1. Execute Teleport
        if self.teleport_queued:
            beat_reward += self._execute_teleport()
            self.teleport_queued = False
        
        # 2. Update Platforms
        self._update_platforms()
        
        # 3. Update Enemies
        self._update_enemies()
        
        # 4. Check State
        if not self.is_teleporting:
            beat_reward += self._check_state()

        return beat_reward

    def _execute_teleport(self):
        reward = 0
        target_platform = self.platforms[self.player_target_platform_idx]

        if not target_platform.active:
            self.game_over = True # Fell onto inactive platform
            self._create_particle_burst(self.player_pos, self.COLOR_ENEMY, 50)
            return reward
        
        # Successful teleport
        reward += 0.1
        self.score += 1
        
        self.is_teleporting = True
        self.teleport_anim_timer = self.BEAT_DURATION // 2
        self.teleport_start_pos = self.player_pos
        self.teleport_end_pos = target_platform.pos
        
        self.player_platform_idx = self.player_target_platform_idx
        
        if target_platform.is_chain_trigger:
            reward += 1
            self.score += 10
            self._trigger_chain_reaction(self.player_platform_idx)
        else:
            pass
        
        self._create_particle_burst(self.teleport_start_pos, self.COLOR_PLAYER, 20, is_implosion=True)
        return reward

    def _check_state(self):
        reward = 0
        # Check for fall
        current_platform = self.platforms[self.player_platform_idx]
        if not current_platform.active:
            self.game_over = True
            self._create_particle_burst(self.player_pos, self.COLOR_ENEMY, 50)
            return reward

        # Check for enemy collision
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_RADIUS, self.player_pos[1] - self.PLAYER_RADIUS, self.PLAYER_RADIUS*2, self.PLAYER_RADIUS*2)
        for enemy in self.enemies:
            enemy_rect = pygame.Rect(enemy['pos'][0] - self.ENEMY_RADIUS, enemy['pos'][1] - self.ENEMY_RADIUS, self.ENEMY_RADIUS*2, self.ENEMY_RADIUS*2)
            if player_rect.colliderect(enemy_rect):
                self.game_over = True
                self._create_particle_burst(self.player_pos, self.COLOR_ENEMY, 50)
                return reward
            
            # Near miss penalty
            dist = math.hypot(self.player_pos[0] - enemy['pos'][0], self.player_pos[1] - enemy['pos'][1])
            if dist < self.PLATFORM_RADIUS * 2.5:
                reward -= 5
        
        # Check for win
        if current_platform.is_exit:
            self.game_over = True
            self.player_on_exit = True
            self.level += 1
            if (self.level - 1) % 3 == 0:
                self.unlocked_visuals = min(self.unlocked_visuals + 1, 2)

        return reward

    def _update_platforms(self):
        beat_num = self.steps // self.BEAT_DURATION
        for i, p in enumerate(self.platforms):
            if p.chain_timer > 0:
                p.chain_timer -= 1
                if p.chain_timer == 0:
                    p.active = False
            else:
                p.active = (beat_num + i) % 3 != 0
        
        # Anti-softlock
        active_platforms = [p for p in self.platforms if p.active]
        if not active_platforms:
            self.np_random.choice(self.platforms).active = True

        # Ensure start/exit are always active
        self.platforms[self.start_platform_idx].active = True
        self.platforms[self.exit_platform_idx].active = True

    def _update_enemies(self):
        for enemy in self.enemies:
            target_pos = self.platforms[enemy['path'][enemy['target_idx']]].pos
            dx, dy = target_pos[0] - enemy['pos'][0], target_pos[1] - enemy['pos'][1]
            dist = math.hypot(dx, dy)
            
            if dist < enemy['speed']:
                enemy['pos'] = target_pos
                enemy['target_idx'] = (enemy['target_idx'] + 1) % len(enemy['path'])
            else:
                enemy['pos'] = (enemy['pos'][0] + dx/dist * enemy['speed'], 
                                enemy['pos'][1] + dy/dist * enemy['speed'])

    def _update_animations(self):
        if self.is_teleporting:
            self.teleport_anim_timer -= 1
            if self.teleport_anim_timer <= 0:
                self.is_teleporting = False
                self.player_pos = self.teleport_end_pos
                self._create_particle_burst(self.player_pos, self.COLOR_PLAYER, 30)
                # Check state immediately after landing
                reward = self._check_state()
                self.score += reward # Assuming reward is just score change here
            else:
                progress = 1.0 - (self.teleport_anim_timer / (self.BEAT_DURATION / 2))
                progress = 1 - pow(1 - progress, 4) # Ease out quart
                self.player_pos = (
                    self.teleport_start_pos[0] + (self.teleport_end_pos[0] - self.teleport_start_pos[0]) * progress,
                    self.teleport_start_pos[1] + (self.teleport_end_pos[1] - self.teleport_start_pos[1]) * progress
                )

    def _trigger_chain_reaction(self, trigger_idx):
        trigger_pos = self.platforms[trigger_idx].pos
        for i, p in enumerate(self.platforms):
            if i == trigger_idx: continue
            dist = math.hypot(p.pos[0] - trigger_pos[0], p.pos[1] - trigger_pos[1])
            if dist < self.PLATFORM_RADIUS * 4:
                p.chain_timer = self.BEAT_DURATION // 2 # Deactivate for half a beat
                self._create_particle_burst(p.pos, self.COLOR_PLATFORM_CHAIN, 10)

    def _generate_level(self):
        self.platforms = []
        self.enemies = []
        
        num_platforms = 10 + self.level
        num_enemies = 1 + self.level // 2
        enemy_speed = 2.0 + self.level * 0.05
        
        while True:
            self.platforms.clear()
            
            # Generate platforms
            platform_locs = set()
            while len(platform_locs) < num_platforms:
                x = self.np_random.integers(self.PLATFORM_RADIUS*2, self.WIDTH - self.PLATFORM_RADIUS*2)
                y = self.np_random.integers(self.PLATFORM_RADIUS*2, self.HEIGHT - self.PLATFORM_RADIUS*2)
                
                # Ensure platforms are not too close
                too_close = False
                for px, py in platform_locs:
                    if math.hypot(x-px, y-py) < self.PLATFORM_RADIUS * 3:
                        too_close = True
                        break
                if not too_close:
                    platform_locs.add((x, y))
            
            for i, pos in enumerate(platform_locs):
                self.platforms.append(Platform(pos, self.unlocked_visuals))

            # Assign start/exit
            self.start_platform_idx = self.np_random.integers(num_platforms)
            farthest_dist = -1
            self.exit_platform_idx = -1
            start_pos = self.platforms[self.start_platform_idx].pos
            for i, p in enumerate(self.platforms):
                dist = math.hypot(p.pos[0] - start_pos[0], p.pos[1] - start_pos[1])
                if dist > farthest_dist:
                    farthest_dist = dist
                    self.exit_platform_idx = i
            
            self.platforms[self.exit_platform_idx].is_exit = True
            
            # Check connectivity
            adj = [[] for _ in range(num_platforms)]
            for i in range(num_platforms):
                for j in range(i + 1, num_platforms):
                    dist = math.hypot(self.platforms[i].pos[0] - self.platforms[j].pos[0], self.platforms[i].pos[1] - self.platforms[j].pos[1])
                    if dist < self.TELEPORT_DIST_MAX:
                        adj[i].append(j)
                        adj[j].append(i)
            
            q = [self.start_platform_idx]
            visited = {self.start_platform_idx}
            while q:
                u = q.pop(0)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        q.append(v)
            
            if self.exit_platform_idx in visited:
                break # Solvable level generated

        # Assign chain reaction platforms
        num_chain = self.np_random.integers(1, 3)
        possible_chain_indices = [i for i in range(num_platforms) if i != self.start_platform_idx and i != self.exit_platform_idx]
        if possible_chain_indices:
            chain_indices = self.np_random.choice(possible_chain_indices, size=min(num_chain, len(possible_chain_indices)), replace=False)
            for i in chain_indices:
                self.platforms[i].is_chain_trigger = True

        # Generate enemies, ensuring they don't spawn on or path through the start platform
        possible_enemy_platforms = [i for i in range(num_platforms) if i != self.start_platform_idx]
        if len(possible_enemy_platforms) >= 2:
            for _ in range(num_enemies):
                path_indices = self.np_random.choice(possible_enemy_platforms, size=2, replace=False).tolist()
                start_pos_enemy = self.platforms[path_indices[0]].pos
                self.enemies.append({'pos': start_pos_enemy, 'path': path_indices, 'target_idx': 1, 'speed': enemy_speed})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background grid
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Connections between platforms
        for i, p1 in enumerate(self.platforms):
            for j, p2 in enumerate(self.platforms):
                if i >= j: continue
                dist = math.hypot(p1.pos[0]-p2.pos[0], p1.pos[1]-p2.pos[1])
                if dist < self.TELEPORT_DIST_MAX:
                    alpha = int(max(0, 1 - (dist / self.TELEPORT_DIST_MAX)) * 50)
                    pygame.draw.aaline(self.screen, self.COLOR_GRID + (alpha,), p1.pos, p2.pos)

        # Platforms
        beat_progress = self.beat_timer / self.BEAT_DURATION
        for p in self.platforms:
            p.draw(self.screen, beat_progress)

        # Enemies
        for enemy in self.enemies:
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ENEMY_RADIUS, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ENEMY_RADIUS, self.COLOR_ENEMY)

        # Marker and arrow
        if not self.is_teleporting:
            marker_pos = self.platforms[self.player_target_platform_idx].pos
            pygame.gfxdraw.aacircle(self.screen, int(marker_pos[0]), int(marker_pos[1]), self.PLATFORM_RADIUS + 5, self.COLOR_MARKER)
            
            if self.player_target_platform_idx != self.player_platform_idx:
                start_pos = self.platforms[self.player_platform_idx].pos
                end_pos = self.platforms[self.player_target_platform_idx].pos
                dx, dy = end_pos[0] - start_pos[0], end_pos[1] - start_pos[1]
                dist = math.hypot(dx, dy)
                if dist > 0:
                    dx /= dist
                    dy /= dist
                    arrow_start = (start_pos[0] + dx * (self.PLAYER_RADIUS + 2), start_pos[1] + dy * (self.PLAYER_RADIUS + 2))
                    arrow_end = (end_pos[0] - dx * (self.PLATFORM_RADIUS + 7), end_pos[1] - dy * (self.PLATFORM_RADIUS + 7))
                    pygame.draw.aaline(self.screen, self.COLOR_MARKER, arrow_start, arrow_end)

        # Particles
        for p in self.particles:
            p.draw(self.screen)

        # Player
        if not (self.game_over and not self.player_on_exit):
            pos = (int(self.player_pos[0]), int(self.player_pos[1]))
            
            # Glow effect
            glow_radius = int(self.PLAYER_RADIUS * (1.5 + 0.5 * math.sin(self.steps * 0.2)))
            glow_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, self.COLOR_PLAYER_GLOW)
            self.screen.blit(glow_surf, (pos[0]-glow_radius, pos[1]-glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
            
            # Player circle
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score and Level
        score_text = self.font_main.render(f"SCORE: {self.score + self.persistent_score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        level_text = self.font_main.render(f"LEVEL: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.WIDTH - level_text.get_width() - 10, 10))
        
        # Beat indicator around player
        beat_progress = self.beat_timer / self.BEAT_DURATION
        pulse_radius = int(self.PLAYER_RADIUS + 10 * (1-beat_progress))
        pulse_alpha = int(200 * (1-beat_progress)**2)
        if pulse_alpha > 0 and not self.is_teleporting:
            pygame.gfxdraw.aacircle(self.screen, int(self.player_pos[0]), int(self.player_pos[1]), pulse_radius, self.COLOR_PLAYER + (pulse_alpha,))

        if self.game_over:
            msg = "LEVEL COMPLETE" if self.player_on_exit else "FAILURE"
            color = self.COLOR_PLATFORM_ACTIVE if self.player_on_exit else self.COLOR_ENEMY
            end_text = self.font_big.render(msg, True, color)
            self.screen.blit(end_text, end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2)))

    def _get_info(self):
        return {"score": self.score + self.persistent_score, "steps": self.steps, "level": self.level}

    def _create_particle_burst(self, pos, color, count, is_implosion=False):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            if is_implosion:
                vel = [-v for v in vel]
            self.particles.append(Particle(pos, vel, color, self.PARTICLE_LIFESPAN))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]


class Platform:
    def __init__(self, pos, visual_style=0):
        self.pos = pos
        self.active = True
        self.is_exit = False
        self.is_chain_trigger = False
        self.chain_timer = 0
        self.visual_style = visual_style # 0=circle, 1=square, 2=hexagon
        self.radius = GameEnv.PLATFORM_RADIUS

    def draw(self, surface, beat_progress):
        pos_int = (int(self.pos[0]), int(self.pos[1]))
        
        if self.is_exit:
            color = GameEnv.COLOR_EXIT
        elif self.is_chain_trigger:
            color = GameEnv.COLOR_PLATFORM_CHAIN
        elif self.active:
            color = GameEnv.COLOR_PLATFORM_ACTIVE
        else:
            color = GameEnv.COLOR_PLATFORM_INACTIVE

        # Pulse effect for active platforms
        radius = self.radius
        if self.active:
            pulse = math.sin(beat_progress * math.pi * 2) * 2
            radius = int(self.radius + pulse)

        if self.visual_style == 0: # Circle
            pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], radius, color)
            pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], radius, color)
        elif self.visual_style == 1: # Square
            rect = pygame.Rect(pos_int[0] - radius, pos_int[1] - radius, radius*2, radius*2)
            pygame.draw.rect(surface, color, rect)
        elif self.visual_style == 2: # Hexagon
            points = []
            for i in range(6):
                angle = math.pi / 3 * i
                points.append((pos_int[0] + radius * math.cos(angle), pos_int[1] + radius * math.sin(angle)))
            pygame.gfxdraw.filled_polygon(surface, points, color)
            pygame.gfxdraw.aapolygon(surface, points, color)

class Particle:
    def __init__(self, pos, vel, color, lifespan):
        self.pos = list(pos)
        self.vel = list(vel)
        self.color = color
        self.lifespan = lifespan
        self.max_lifespan = lifespan

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.vel[0] *= 0.95 # Damping
        self.vel[1] *= 0.95
        self.lifespan -= 1
        return self.lifespan > 0

    def draw(self, surface):
        progress = self.lifespan / self.max_lifespan
        radius = int(progress * 4)
        if radius > 0:
            alpha = int(progress * 255)
            # Create a temporary surface for alpha blending if color doesn't have alpha
            if len(self.color) == 3:
                temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, self.color + (alpha,), (radius, radius), radius)
                surface.blit(temp_surf, (int(self.pos[0]) - radius, int(self.pos[1]) - radius))
            else: # Color already has alpha
                pygame.draw.circle(surface, self.color[:3] + (alpha,), (int(self.pos[0]), int(self.pos[1])), radius)


if __name__ == '__main__':
    # Un-comment the line below to run with a display
    # os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    
    env = GameEnv(render_mode='rgb_array')
    obs, info = env.reset()
    
    # Manual play loop
    running = True
    terminated = False
    
    # Pygame setup for display
    # This will fail if SDL_VIDEODRIVER is "dummy", but is useful for local testing
    try:
        display_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Rhythmic Stealth Platformer")
        clock = pygame.time.Clock()
        has_display = True
    except pygame.error:
        print("No display available. Running headlessly.")
        has_display = False

    action = env.action_space.sample()
    action[0] = 0 # Start with no-op movement
    action[1] = 0 # Start with space released
    
    while running:
        if has_display:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            
            mov = 0
            if keys[pygame.K_UP] or keys[pygame.K_w]: mov = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: mov = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: mov = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: mov = 4
            
            space = 1 if keys[pygame.K_SPACE] else 0
            shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = [mov, space, shift]
            
            if terminated:
                # Wait for a key press to reset after game over
                if any(keys):
                    obs, info = env.reset()
                    terminated = False
            else:
                obs, reward, terminated, truncated, info = env.step(action)

            # Display the observation
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            display_screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            clock.tick(GameEnv.FPS)
        else: # Headless loop
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                print(f"Episode finished. Score: {info['score']}, Steps: {info['steps']}")
                obs, info = env.reset()


    if has_display:
        pygame.quit()