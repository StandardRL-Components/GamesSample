import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:28:44.730750
# Source Brief: brief_01116.md
# Brief Index: 1116
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your network core from incoming projectiles by placing temporary shields. "
        "Upgrade your abilities to survive all waves."
    )
    user_guide = (
        "Controls: Use ↑↓ to rotate shield placement. Press space to deploy a shield. "
        "Hold shift to open/close the skill tree, then use ↑↓ to navigate and space to purchase."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 4000
    TOTAL_WAVES = 20

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_CORE = (0, 255, 150)
    COLOR_CORE_DAMAGED = (255, 255, 0)
    COLOR_CORE_CRITICAL = (255, 50, 50)
    COLOR_SHIELD = (0, 255, 200)
    COLOR_PROJECTILE = (255, 80, 80)
    COLOR_PARTICLE = (255, 200, 150)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_ACCENT = (100, 100, 255)
    COLOR_UI_PANEL = (20, 30, 60, 220)
    COLOR_UI_DISABLED = (100, 100, 120)

    # Game Parameters
    CORE_RADIUS = 30
    SHIELD_PLACEMENT_RADIUS = 80
    SHIELD_THICKNESS = 5
    SHIELD_ARC_LENGTH = math.pi / 2  # 90 degrees
    ROTATION_SPEED = 0.1  # Radians per step

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_medium = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # --- Game State Initialization ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.core_health = 0.0
        self.income = 0.0
        self.current_wave = 0
        self.wave_timer = 0
        self.projectiles_this_wave = 0
        self.projectiles_spawned = 0
        self.projectiles = []
        self.shields = []
        self.particles = []
        self.placement_angle = 0.0
        self.previous_space_held = False
        self.previous_shift_held = False
        self.skill_tree_open = False
        self.selected_skill_idx = 0
        self.skill_list = []
        self.skills = {}

        # self.reset() is called by the wrapper, no need to call it here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.core_health = 100.0
        self.income = 100.0
        self.current_wave = 1
        self.wave_timer = 0
        
        self._setup_wave_parameters()

        self.projectiles.clear()
        self.shields.clear()
        self.particles.clear()
        
        self.placement_angle = 0.0
        self.previous_space_held = False
        self.previous_shift_held = False
        self.skill_tree_open = False
        self.selected_skill_idx = 0
        
        self._initialize_skills()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.01  # Small penalty for every time step
        self.steps += 1

        if not self.game_over:
            self._handle_input(action)
            reward += self._update_game_state()

        terminated = self._check_termination()
        if terminated and not self.game_over:
            if self.win:
                reward += 100
            else:
                reward += -100
            self.game_over = True
            
        self.score += reward

        # Truncated is always False for this environment
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _initialize_skills(self):
        self.skills = {
            "shield_duration": {"name": "Shield Duration", "level": 1, "cost": 50, "value": 4.0, "increment": 0.5, "max_level": 10},
            "income_rate": {"name": "Income Rate", "level": 1, "cost": 75, "value": 10.0, "increment": 5.0, "max_level": 10},
            "shield_arc": {"name": "Shield Arc", "level": 1, "cost": 150, "value": self.SHIELD_ARC_LENGTH, "increment": math.pi / 12, "max_level": 5},
            "core_repair": {"name": "Repair Core (+10 HP)", "level": 1, "cost": 200, "value": 10, "increment": 0, "max_level": 99},
        }
        self.skill_list = list(self.skills.keys())

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Shift Key: Toggle Skill Tree ---
        if shift_held and not self.previous_shift_held:
            self.skill_tree_open = not self.skill_tree_open
            # SFX: UI_open_close

        if self.skill_tree_open:
            # --- Skill Tree Navigation ---
            if movement == 1: # Up
                self.selected_skill_idx = max(0, self.selected_skill_idx - 1)
            elif movement == 2: # Down
                self.selected_skill_idx = min(len(self.skill_list) - 1, self.selected_skill_idx + 1)
            
            if space_held and not self.previous_space_held:
                self._buy_skill()
        else:
            # --- Shield Placement Controls ---
            if movement == 1: # Up (Rotate CW)
                self.placement_angle -= self.ROTATION_SPEED
            elif movement == 2: # Down (Rotate CCW)
                self.placement_angle += self.ROTATION_SPEED
            
            # Keep angle in [0, 2*pi]
            self.placement_angle %= (2 * math.pi)

            if space_held and not self.previous_space_held:
                self._deploy_shield()

        self.previous_space_held = space_held
        self.previous_shift_held = shift_held

    def _update_game_state(self):
        frame_reward = 0
        
        # --- Update Income ---
        self.income += self.skills["income_rate"]["value"] / self.FPS

        # --- Update Shields ---
        self.shields[:] = [s for s in self.shields if s['duration'] > 0]
        for shield in self.shields:
            shield['duration'] -= 1 / self.FPS

        # --- Update Particles ---
        self.particles[:] = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['size'] = max(0, p['size'] - 0.1)

        # --- Update Projectiles ---
        for proj in self.projectiles:
            proj['pos'] += proj['vel']

        # --- Collision Detection ---
        projectiles_to_remove = set()
        for i, proj in enumerate(self.projectiles):
            if i in projectiles_to_remove:
                continue

            # Projectile-Shield Collision
            collided_with_shield = False
            for shield in self.shields:
                if self._check_projectile_shield_collision(proj, shield):
                    self._deflect_projectile(proj, shield)
                    self._create_particles(proj['pos'], 15, self.COLOR_SHIELD, 2)
                    frame_reward += 0.1
                    collided_with_shield = True
                    # SFX: shield_deflect
                    break
            if collided_with_shield:
                continue

            # Projectile-Core Collision
            dist_to_core = proj['pos'].distance_to(pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2))
            if dist_to_core < self.CORE_RADIUS:
                self.core_health -= proj['damage']
                self.core_health = max(0, self.core_health)
                projectiles_to_remove.add(i)
                self._create_particles(proj['pos'], 30, self.COLOR_CORE_CRITICAL, 4)
                # SFX: core_damage
            
            # Projectile-Boundary Collision (Remove if off-screen)
            elif not self.screen.get_rect().collidepoint(proj['pos']):
                projectiles_to_remove.add(i)

        if projectiles_to_remove:
            self.projectiles = [p for i, p in enumerate(self.projectiles) if i not in projectiles_to_remove]
        
        # --- Wave Management ---
        if not self.skill_tree_open:
            self.wave_timer += 1
        if self.projectiles_spawned < self.projectiles_this_wave and self.wave_timer > self.spawn_interval:
            self._spawn_projectile()
            self.wave_timer = 0
        
        # Check for wave completion
        if self.projectiles_spawned == self.projectiles_this_wave and not self.projectiles:
            self.current_wave += 1
            frame_reward += 1.0
            # SFX: wave_complete
            if self.current_wave > self.TOTAL_WAVES:
                self.win = True
            else:
                self._setup_wave_parameters()

        return frame_reward

    def _setup_wave_parameters(self):
        self.projectiles_spawned = 0
        self.wave_timer = 0
        
        base_projectiles = 3
        projectiles_per_5_waves = (self.current_wave - 1) // 5
        self.projectiles_this_wave = base_projectiles + self.current_wave + projectiles_per_5_waves
        
        self.spawn_interval = max(10, 60 - self.current_wave * 2)
        
        self.projectile_speed = 1.5 + self.current_wave * 0.2
        self.projectile_damage = 5 + self.current_wave * 0.5

    def _spawn_projectile(self):
        self.projectiles_spawned += 1
        
        edge = self.np_random.integers(4)
        if edge == 0: # Top
            pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), -10)
        elif edge == 1: # Bottom
            pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 10)
        elif edge == 2: # Left
            pos = pygame.Vector2(-10, self.np_random.uniform(0, self.HEIGHT))
        else: # Right
            pos = pygame.Vector2(self.WIDTH + 10, self.np_random.uniform(0, self.HEIGHT))
        
        core_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        direction = (core_pos - pos).normalize()
        vel = direction * self.projectile_speed

        self.projectiles.append({'pos': pos, 'vel': vel, 'damage': self.projectile_damage})
        # SFX: projectile_spawn

    def _deploy_shield(self):
        shield_duration = self.skills["shield_duration"]["value"]
        
        self.shields.append({
            'angle': self.placement_angle,
            'duration': shield_duration,
            'max_duration': shield_duration
        })
        # SFX: shield_deploy

    def _buy_skill(self):
        skill_key = self.skill_list[self.selected_skill_idx]
        skill = self.skills[skill_key]

        if skill['level'] < skill['max_level'] and self.income >= skill['cost']:
            self.income -= skill['cost']
            skill['level'] += 1
            
            if skill_key == "core_repair":
                self.core_health = min(100, self.core_health + skill['value'])
                skill['cost'] *= 1.5 # Repair cost increases
            else:
                skill['value'] += skill['increment']
                skill['cost'] *= 1.5 # Standard cost increase
            
            skill['cost'] = int(skill['cost'])
            # SFX: upgrade_purchased
        else:
            # SFX: upgrade_failed
            pass

    def _check_projectile_shield_collision(self, proj, shield):
        core_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        dist_to_core = proj['pos'].distance_to(core_pos)
        
        # Fast check: is projectile within the shield's radial zone?
        if not (self.SHIELD_PLACEMENT_RADIUS - self.SHIELD_THICKNESS < dist_to_core < self.SHIELD_PLACEMENT_RADIUS + self.SHIELD_THICKNESS):
            return False
            
        # Slower check: is projectile within the shield's angular arc?
        proj_angle = math.atan2(core_pos.y - proj['pos'].y, proj['pos'].x - core_pos.x)
        proj_angle = (proj_angle + 2 * math.pi) % (2 * math.pi) # Normalize to [0, 2pi]
        
        shield_arc = self.skills["shield_arc"]["value"]
        start_angle = (shield['angle'] - shield_arc / 2) % (2 * math.pi)
        end_angle = (shield['angle'] + shield_arc / 2) % (2 * math.pi)

        # Handle angle wrapping
        if start_angle <= end_angle:
            return start_angle <= proj_angle <= end_angle
        else: # Wraps around 0
            return proj_angle >= start_angle or proj_angle <= end_angle

    def _deflect_projectile(self, proj, shield):
        core_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        
        # The reflection normal is the vector from the core to the projectile
        normal = (proj['pos'] - core_pos).normalize()
        
        # Reflect the velocity vector
        proj['vel'] = proj['vel'].reflect(normal)

    def _create_particles(self, pos, count, color, speed_factor):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, speed_factor)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifetime': self.np_random.integers(10, 25),
                'size': self.np_random.uniform(1, 4),
                'color': color
            })

    def _check_termination(self):
        return self.core_health <= 0 or self.win

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "core_health": self.core_health,
            "income": self.income,
        }

    def _render_game(self):
        # --- Draw Particles ---
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], p['size'])

        # --- Draw Core ---
        core_pos = (self.WIDTH // 2, self.HEIGHT // 2)
        health_percent = self.core_health / 100.0
        if health_percent > 0.6:
            core_color = self.COLOR_CORE
        elif health_percent > 0.3:
            core_color = self.COLOR_CORE_DAMAGED
        else:
            core_color = self.COLOR_CORE_CRITICAL
        
        # Glow effect
        pulse = abs(math.sin(self.steps * 0.05))
        glow_radius = self.CORE_RADIUS + 5 + pulse * 5
        glow_alpha = int(50 + pulse * 50)
        pygame.gfxdraw.filled_circle(self.screen, core_pos[0], core_pos[1], int(glow_radius), (*core_color, glow_alpha))
        pygame.gfxdraw.aacircle(self.screen, core_pos[0], core_pos[1], int(glow_radius), (*core_color, glow_alpha))
        
        # Main core circle
        pygame.gfxdraw.filled_circle(self.screen, core_pos[0], core_pos[1], self.CORE_RADIUS, core_color)
        pygame.gfxdraw.aacircle(self.screen, core_pos[0], core_pos[1], self.CORE_RADIUS, core_color)

        # --- Draw Shield Placement Indicator ---
        if not self.skill_tree_open:
            shield_arc = self.skills["shield_arc"]["value"]
            start_angle = self.placement_angle - shield_arc / 2
            end_angle = self.placement_angle + shield_arc / 2
            indicator_color = (*self.COLOR_SHIELD, 100)
            self._draw_arc(self.screen, indicator_color, core_pos, self.SHIELD_PLACEMENT_RADIUS, start_angle, end_angle, 3)

        # --- Draw Active Shields ---
        for shield in self.shields:
            alpha = int(max(0, min(1, shield['duration'] / 2.0)) * 255)
            shield_color = (*self.COLOR_SHIELD, alpha)
            
            shield_arc = self.skills["shield_arc"]["value"]
            start_angle = shield['angle'] - shield_arc / 2
            end_angle = shield['angle'] + shield_arc / 2
            
            # Glow
            self._draw_arc(self.screen, (*self.COLOR_SHIELD, alpha // 4), core_pos, self.SHIELD_PLACEMENT_RADIUS, start_angle, end_angle, self.SHIELD_THICKNESS + 6)
            # Main arc
            self._draw_arc(self.screen, shield_color, core_pos, self.SHIELD_PLACEMENT_RADIUS, start_angle, end_angle, self.SHIELD_THICKNESS)

        # --- Draw Projectiles ---
        for proj in self.projectiles:
            start_pos = proj['pos']
            end_pos = proj['pos'] - proj['vel'].normalize() * 10
            # Glow
            pygame.draw.line(self.screen, (*self.COLOR_PROJECTILE, 100), start_pos, end_pos, 6)
            # Main line
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE, start_pos, end_pos, 3)

    def _render_ui(self):
        # --- Top Bar Info ---
        wave_text = self.font_small.render(f"WAVE: {self.current_wave}/{self.TOTAL_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (10, 10))

        income_text = self.font_small.render(f"INCOME: {int(self.income)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(income_text, (self.WIDTH - income_text.get_width() - 10, 10))

        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, 10))
        
        # --- Core Health Bar ---
        health_bar_width = 200
        health_bar_height = 15
        health_bar_x = self.WIDTH // 2 - health_bar_width // 2
        health_bar_y = self.HEIGHT - health_bar_height - 10
        
        health_percent = self.core_health / 100.0
        current_health_width = int(health_bar_width * health_percent)
        
        pygame.draw.rect(self.screen, self.COLOR_CORE_CRITICAL, (health_bar_x, health_bar_y, health_bar_width, health_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_CORE, (health_bar_x, health_bar_y, current_health_width, health_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (health_bar_x, health_bar_y, health_bar_width, health_bar_height), 1)

        # --- Skill Tree Panel ---
        if self.skill_tree_open:
            self._render_skill_tree()

        # --- Game Over / Win Screen ---
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN" if self.win else "GAME OVER"
            color = self.COLOR_CORE if self.win else self.COLOR_CORE_CRITICAL
            
            end_text = self.font_large.render(message, True, color)
            self.screen.blit(end_text, (self.WIDTH // 2 - end_text.get_width() // 2, self.HEIGHT // 2 - end_text.get_height() // 2 - 20))
            
            score_text = self.font_medium.render(f"Final Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
            self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, self.HEIGHT // 2 + 30))

    def _render_skill_tree(self):
        panel_width, panel_height = 400, 250
        panel_x = self.WIDTH // 2 - panel_width // 2
        panel_y = self.HEIGHT // 2 - panel_height // 2
        
        # Draw panel background
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.gfxdraw.box(self.screen, panel_rect, self.COLOR_UI_PANEL)
        pygame.draw.rect(self.screen, self.COLOR_UI_ACCENT, panel_rect, 2)

        # Title
        title_text = self.font_medium.render("SKILL TREE", True, self.COLOR_UI_ACCENT)
        self.screen.blit(title_text, (panel_x + panel_width // 2 - title_text.get_width() // 2, panel_y + 10))

        # Render skills
        y_offset = panel_y + 50
        for i, skill_key in enumerate(self.skill_list):
            skill = self.skills[skill_key]
            
            is_selected = (i == self.selected_skill_idx)
            can_afford = self.income >= skill['cost']
            is_maxed = skill['level'] >= skill['max_level']

            # Determine text color
            if is_selected:
                color = self.COLOR_UI_ACCENT
            elif is_maxed:
                color = self.COLOR_CORE
            elif not can_afford:
                color = self.COLOR_UI_DISABLED
            else:
                color = self.COLOR_UI_TEXT

            # Draw selection highlight
            if is_selected:
                highlight_rect = pygame.Rect(panel_x + 10, y_offset - 2, panel_width - 20, 24)
                pygame.draw.rect(self.screen, (*self.COLOR_UI_ACCENT, 50), highlight_rect)
                pygame.draw.rect(self.screen, self.COLOR_UI_ACCENT, highlight_rect, 1)

            # Skill name and level
            level_str = "MAX" if is_maxed else f"Lvl {skill['level']}"
            name_text = self.font_small.render(f"{skill['name']} [{level_str}]", True, color)
            self.screen.blit(name_text, (panel_x + 20, y_offset))

            # Cost
            cost_str = "---" if is_maxed else f"Cost: {skill['cost']}"
            cost_text = self.font_small.render(cost_str, True, color)
            self.screen.blit(cost_text, (panel_x + panel_width - cost_text.get_width() - 20, y_offset))

            y_offset += 30

    def _draw_arc(self, surface, color, center, radius, start_angle, end_angle, width):
        points = []
        num_steps = int(abs(end_angle - start_angle) * 30) # Quality control
        if num_steps < 2: return
        
        for i in range(num_steps + 1):
            angle = start_angle + (end_angle - start_angle) * i / num_steps
            x = center[0] + radius * math.cos(angle)
            y = center[1] - radius * math.sin(angle)
            points.append((x, y))
        
        if len(points) > 1:
            pygame.draw.lines(surface, color, False, points, width)

    def validate_implementation(self):
        # This was part of the original code for self-checking and can be removed or kept.
        # It's not part of the standard Gymnasium API.
        pass

if __name__ == '__main__':
    # This block allows you to play the game manually
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Network Core Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Action state
    movement = 0
    space = 0
    shift = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE:
                    space = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    space = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        else:
            movement = 0
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Wave: {info['wave']}")
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)

    pygame.quit()